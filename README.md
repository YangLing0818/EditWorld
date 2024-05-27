# EditWorld: Simulating World Dynamics for Instruction-Following Image Editing

## Overview

This repository contains the official implementation of our [EditWorld](https://arxiv.org/abs/2405.14785). In this work, we introduce a new task namely **world-instructed image editing**, which defines and categorizes the instructions grounded by various world scenarios. We curate a new image editing dataset with world instructions using a set of large pretrained models (e.g., GPT-3.5, Video-LLava and SDXL). We also propose a new post-edit method for world-instructed image editing.

### World Instruction *vs*. Traditional Instruction
![first_img](assets/first_img.jpg)

### Generated Results of Our EditWorld:
![sample1](assets/generation_samples.jpg)



## Planning 
  - [√] Providing full pipeline of text-to-image generation for EditWorld dataset.
  - [√] Releasing evaluation dataset.
  - [ ] Releasing Checkpoints.
  - [ ] Releasing training and post-edit code.

## Codebase

### Text-to-image generation branch

Firstly, we employ GPT-3.5 to provide textual quadruples:

```shell
python gpt_script/text_img_gen_aigcbest_full.py --define_json gpt_script/define_sample_history/define_sample.json --output_path gpt_script/gen_sample_history/ --output_json text_gen.json
```

Then, we transform the text prompt provided by GPT into dict:

```shell
python tools/deal_text2json.py --input_json gpt_script/gen_sample_history/text_gen.json --output_json text_gen_full.json
```

Finally, we obtain the input-instruct-output triples based on the generated textual quadruples:

```shell
python t2i_branch_base.py --text_json text_gen_full.json --save_path datasets/editworld/generated_img/
```

It is worth noting that `t2i_branch_base.py` is the fast and basic version for text-to-image generation branch, we will improve this part in the future.

### Video branch

Path `video_script` contains the code for downloading videos from the [InternVid](https://huggingface.co/datasets/OpenGVLab/InternVid).

## Dataset

### Dataset structure

To obtain the training dataset file `train.json`, utilize the script located at `tools/obtain_datasetjson.py`. The dataset is organized in the following structure:

```css
datasets/
├── editworld/
│   ├── generated_img/
│   │   ├── group_0/
│   │   │   ├── sample0_ori.png
│   │   │   ├── sample0_tar.png
│   │   │   ...
│   │   │   └── img_txt.json
│   │   └── group_1/
│   │   ...
│   ├── video_img/
│   │   ├── group_0/
│   │   │   ├── sample0_ori.png
│   │   │   ├── sample0_tar.png
│   │   │   ...
│   │   │   └── img_txt.json
│   │   └── group_1/
│   │   ...
│   └── human_select_img/
│       ├── group_0/
│       │   ├── sample0_ori.png
│       │   ├── sample0_tar.png
│       │   ...
│       │   └── img_txt.json
│       └── group_1/
│       ...
└── train.json
```

### Evaluation dataset link

Our evaluation dataset is available at [editworld_test](https://drive.google.com/drive/u/1/folders/1ReuBMCNiCIVT-pC6YnM9Rv2irJUMFfh7).


## Quantitative Comparison of CLIP Score and MLLM Score

IP2P: [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix); MB: [MagicBrush](https://github.com/OSU-NLP-Group/MagicBrush). Bold results are the best.

### CLIP Score of Text-to-image Branch

| Category           | IP2P     | MB       | Editworld | w/o post-edit   |
|--------------------|----------|----------|-----------|-----------------|
| Long-Term          | 0.2140   | 0.1870   | 0.2244    | **0.2294**      |
| Physical-Trans     | 0.2186   | 0.2101   | 0.2385    | **0.2467**      |
| Implicit-Logic     | 0.2390   | 0.2432   | **0.2542**| 0.2440        |
| Story-Type         | 0.2063   | 0.2070   | **0.2534**| 0.2354        |
| Real-to-Virtual    | 0.2285   | 0.2344   | **0.2524**| 0.2435        |

### CLIP Score of Video Branch

| Category           | IP2P     | MB       | Editworld | w/o post-edit   |
|--------------------|----------|----------|-----------|-----------------|
| Spatial-Trans      | 0.2175   | 0.1997   | **0.2420**| 0.2286        |
| Physical-Trans     | 0.2315   | 0.2278   | 0.2467  | **0.2483**      |
| Story-Type         | 0.2318 | 0.2262   | 0.2365    | **0.2399**      |
| Exaggeration       | 0.2416   | 0.2328   | **0.2443**| 0.2433        |

### MLLM Score of Both Branches

| Category           | IP2P     | MB       | Editworld | w/o post-edit   |
|--------------------|----------|----------|-----------|-----------------|
| Text-to-image      | 0.8763   | 0.8455   | 0.8958  | **0.9060**      |
| Video              | 0.9493   | 0.9715   | **0.9920**| 0.9891        |



## Citation
```
@article{yang2024editworld,
  title={EditWorld: Simulating World Dynamics for Instruction-Following Image Editing},
  author={Yang, Ling and Zeng, Bohan and Liu, Jiaming and Li, Hong and Xu, Minghao and Zhang, Wentao and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2405.14785},
  year={2024}
}
```