import argparse
import os
import re
import json

import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
    )
from PIL import Image
import numpy as np
import random
import cv2

from ip_adapter import IPAdapterXL, IPAdapter
from prompt2prompt.prompt_to_prompt_stable import mask_from_CA, AttentionStore, run_and_display
from clip import clip, tokenize


## The basic judgment version is to use the CLIP score
def CLIP_score(img, text, device='cuda', jit=False):
    """
    img: PIL
    """
    model_path = 'ckpt/ViT-L-14.pt'
    model, preprocess = clip.load(name=model_path, device=device, jit=jit)
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    similarity = F.cosine_similarity(image_features, text_features)
    return similarity


def CLIP_score_I(img1, img2, device='cuda', jit=False):
    """
    img: PIL
    """
    model_path = 'ckpt/ViT-L-14.pt'
    model, preprocess = clip.load(name=model_path, device=device, jit=jit)
    image1 = preprocess(img1).unsqueeze(0).to(device)
    image2 = preprocess(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)
    similarity = F.cosine_similarity(image1_features, image2_features)
    return similarity


def compute_structure_comparison(img1, img2, C3=1e-3):
    """
    img1, img2: PIL
    """
    sigma1_sq = np.var(np.array(img1))
    sigma2_sq = np.var(np.array(img2))

    sigma12 = np.cov(np.array(img1).flatten(), np.array(img2).flatten())[0, 1]

    structure_comparison = (sigma12 + C3) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C3)
    
    return structure_comparison


def obtain_stage1_image(ori_text, key_words, controller, ori_gen_time=1):
    best_clip_score = -999
    for _ in range(ori_gen_time):
        # ori_img_ = pipe_text2img(ori_text).images[0]
        g_cpu = torch.Generator().manual_seed(random.randint(0, 888888))
        ori_img_, _ = run_and_display([ori_text], controller, latent=None, run_baseline=False, generator=g_cpu)
        ori_mask_ = mask_from_CA(controller, res=16, from_where=("up", "down"), prompts=[ori_text], key_words=key_words)
        clip_score = CLIP_score(ori_img_, ori_text)
        if best_clip_score < clip_score:
            best_clip_score = clip_score
            ori_img = ori_img_
            ori_mask = ori_mask_
    print(best_clip_score)
    return ori_img, ori_mask


def obtain_inpainting_results(ori_text, tar_text, ori_img, ori_mask, gen_time_ori=1, gen_time_tar=1):
    best_clip_score, SSIM = -999, -999
    for _ in range(gen_time_ori):
        g_cpu = torch.Generator().manual_seed(random.randint(0, 888888))
        ori_img_ = pipe_inpaint(prompt=ori_text, image=ori_img, mask_image=ori_mask.resize((512, 512))).images[0]
        clip_score = CLIP_score(ori_img_, ori_text)
        if best_clip_score < clip_score:
            best_clip_score = clip_score
            ori_img_1 = ori_img_
    print(best_clip_score)
    for _ in range(gen_time_tar):
        g_cpu = torch.Generator().manual_seed(random.randint(0, 888888))
        tar_img_ = pipe_inpaint(prompt=tar_text, image=ori_img, mask_image=ori_mask.resize((512, 512))).images[0]
        ssim = CLIP_score(tar_img_, tar_text) + CLIP_score_I(tar_img_, ori_img_1) + compute_structure_comparison(tar_img_, ori_img_1)
        if SSIM < ssim:
            SSIM = ssim
            tar_img_1 = tar_img_
    print(SSIM)
    return ori_img_1, tar_img_1


def obtain_ip_tar_imgs(ori_img, tar_img, tar_text, tar_img_canny, ori_mask, gen_time=1):
    # best_clip_I = CLIP_score_I(img1=tar_img, img2=ori_img) + 4*CLIP_score(tar_img, tar_text)
    # tar_img_1 = tar_img
    best_clip_I = -999
    for _ in range(gen_time):
        # tar_img_ = ip_model.generate(pil_image=ori_img, prompt=tar_text, num_samples=1, num_inference_steps=50, seed=42, image=tar_img_canny)[0]
        tar_img_ = ip_model.generate(pil_image=ori_img, prompt=tar_text, num_samples=1, num_inference_steps=50, seed=42,
                                image=tar_img, mask_image=ori_mask.resize(tar_img.size), control_image=tar_img_canny)[0]
        clip_I = CLIP_score_I(img1=tar_img_, img2=ori_img) + 4*CLIP_score(tar_img_, tar_text)
        if best_clip_I < clip_I:
            tar_img_1 = tar_img_
            best_clip_I = clip_I
    return tar_img_1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_json', type=str, default='text_gen_full.json', help="json file of text samples.")
    parser.add_argument('--save_path', type=str, default='datasets/editworld/generated_img/', help="output path for generated results.")
    opt = parser.parse_args()

    base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    image_encoder_xl_path = "ckpt/IP-Adapter/sdxl_models/image_encoder/"
    ip_xl_ckpt = "ckpt/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
    controlnetXL_ckpt = "diffusers/controlnet-canny-sdxl-1.0"
    IP2P_ckpt = 'timbrooks/instruct-pix2pix'
    device='cuda'

    # load ckpt
    ## load SDXL pipeline
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
        use_safetensors=False
    ).to(device)
    pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
        use_safetensors=False
    ).to(device)

    ## load controlnet and ipadapter
    controlnet = ControlNetModel.from_pretrained(
        controlnetXL_ckpt,
        torch_dtype=torch.float16,
        use_safetensors=False
    )
    pipe_control_xl = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
        use_safetensors=False
    ).to(device)
    ip_model = IPAdapterXL(pipe_control_xl, image_encoder_xl_path, ip_xl_ckpt, device)

    # load json
    with open(opt.text_json, 'r') as f:
        texts_dict = json.load(f)
    
    for key in texts_dict.keys():
        sample_id = int(key.replace("sample", ""))
        group_id = sample_id // 100
        index = sample_id % 100

        # output path
        img_save_path = os.path.join(opt.save_path, f'group_{group_id}')
        if os.path.exists(os.path.join(img_save_path, 'img_txt.json')):
            with open(os.path.join(img_save_path, 'img_txt.json')) as f:
                output_dict = json.load(f)
        else:
            os.makedirs(img_save_path, exist_ok=True)
            output_dict = {}

        if f"sample{index}" in output_dict.keys():
            continue

        mid_json_dict = {"instuction": texts_dict[key]["instuction"]}

        # textual prompt
        ori_text = re.sub(r'[^\w\s]', '', texts_dict[key]["original_caption"])
        tar_text = texts_dict[key]["target_cation"]
        key_words = texts_dict[key]["key_words"].split(", ")
        for i in range(len(key_words)):
            wd = key_words[i]
            wd = re.sub(r'[^\w\s]', '', wd)
            if (' ' in wd) and (wd in ori_text):
                wds = wd.split(' ')
                key_words[i] = wds[0]

        # generate stage 1 image
        controller = AttentionStore()
        ori_img, ori_mask = obtain_stage1_image(ori_text, key_words, controller, ori_gen_time=10)

        with torch.no_grad():
            # inpainting
            ori_img, tar_img = obtain_inpainting_results(ori_text=ori_text, tar_text=tar_text, ori_img=ori_img, ori_mask=ori_mask, gen_time_ori=5, gen_time_tar=10)
            ori_img = pipe_img2img(prompt=ori_text + ", realistic", image=ori_img, strength=0.5).images[0]

            # obtain canny of tar_img
            tar_img_canny = np.array(tar_img)
            tar_img_canny = cv2.Canny(tar_img_canny, 100, 200)
            tar_img_canny = tar_img_canny[:, :, None]
            tar_img_canny = np.concatenate([tar_img_canny, tar_img_canny, tar_img_canny], axis=2)
            tar_img_canny = Image.fromarray(tar_img_canny)

            # ip-adapter generation
            tar_img = obtain_ip_tar_imgs(ori_img=ori_img, tar_img=tar_img, tar_text=tar_text, tar_img_canny=tar_img_canny, ori_mask=ori_mask, gen_time=5)
            tar_img = pipe_img2img(prompt=tar_text + ", realistic", image=tar_img, strength=0.5).images[0]

        ori_img.save(os.path.join(img_save_path, f"sample{index}_ori.png"))
        tar_img.save(os.path.join(img_save_path, f"sample{index}_tar.png"))
        mid_json_dict["original_img_path"] = str(os.path.join(img_save_path, f"sample{index}_ori.png"))
        mid_json_dict["target_img_path"] = str(os.path.join(img_save_path, f"sample{index}_tar.png"))
        output_dict[f"sample{index}"] = mid_json_dict
        with open(os.path.join(img_save_path, 'img_txt.json'), 'w') as file:
            json.dump(output_dict, file, indent=4)
