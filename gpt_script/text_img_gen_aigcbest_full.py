import json
import re
import os
import time
import argparse
from aigcbest_api.chat_function import obtain_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--define_json', type=str, default='gpt_script/define_sample_history/define_sample.json', help="json file of text samples.")
    parser.add_argument('--output_path', type=str, default='gpt_script/gen_sample_history/', help="json file of text samples.")
    parser.add_argument('--output_json', type=str, default='text_gen.json', help="json file of text samples.")
    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)

    if os.path.exists(opt.define_json):
        with open(opt.define_json) as f:
            define_dict = json.load(f)

    if os.path.exists(os.path.join(opt.output_path, opt.output_json)):
        with open(os.path.join(opt.output_path, opt.output_json)) as f:
            json_datas = json.load(f)
        i = len(json_datas.keys())
    else:
        json_datas = {}
        i = 0
    for key in define_dict.keys():
        ori_text=define_dict[key]["original_caption"]
        instruct=define_dict[key]["instuction"]
        tar_text=define_dict[key]["target_cation"]
        keywords=define_dict[key]["key_words"]
        init_message = f""" Now you are an "textual prompt creator", Please provide several examples based on real-world physical conditions, \
each example should sequentially include an initial image description, a final image description, image change instructions, and keywords. \
Here's one example: The initial image description is "{ori_text}", the image change instruction is "{instruct}", \
the final image description is "{tar_text}", and the keywords are "{keywords}". Keywords should preferably not be phrases like "paper plane," but rather single words like "apple". \
Please use simple description which is easy for Stable Diffusion model generation.
Please present the examples in the format of "1. {ori_text}; {instruct}; {tar_text}; {keywords}\n2. ...".
"""
        messages = [{"role": "user", "content": init_message}]
        i += 1
        json_datas[f"sample{i}"] = obtain_text(messages)
        messages.append({"role": "assistant", "content": json_datas[f"sample{i}"]})
        for _ in range(1,20):
            messages.append({"role": "user", "content": 'continue'})
            if i % 5 == 0:
                wait_time = 10
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            i += 1
            json_datas[f"sample{i}"] = obtain_text(messages)
            messages.append({"role": "assistant", "content": json_datas[f"sample{i}"]})
            with open(os.path.join(opt.output_path, opt.output_json), 'w') as f:
                json.dump(json_datas, f, indent=4)
