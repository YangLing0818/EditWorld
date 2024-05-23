import os
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='datasets/editworld/', help='json file for further deal.')
    parser.add_argument('--output_json', type=str, default='datasets/editworld/train.json', help='json for dataload.')
    opt = parser.parse_args()

    out_list = []

    for coll_path in os.listdir(opt.input_path):
        samples_num = 0
        data_path = os.path.join(opt.input_path, coll_path)
        if not os.path.isdir(data_path):
            continue
        subpaths = [os.path.join(data_path, name) for name in os.listdir(data_path)
                    if os.path.isdir(os.path.join(data_path, name))]

        for subpath in subpaths:
            if coll_path=="video_img" and (os.path.exists(os.path.join(subpath,'img_txt_ori.json')) is not True):
                continue
            json_file_path = os.path.join(subpath, 'img_txt.json')
            with open(json_file_path, 'r') as f:
                input_dict = json.load(f)
            for key in input_dict.keys():
                if os.path.exists(input_dict[key]["original_img_path"]) and os.path.exists(input_dict[key]["target_img_path"]):
                    out_list.append(input_dict[key])
                    samples_num += 1
        print(f"{coll_path} has {samples_num} image pairs")

    print(f"training samples num is {len(out_list)}")

    with open(opt.output_json, 'w') as file:
        json.dump(out_list, file, indent=4)
