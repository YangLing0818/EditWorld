from pytube import YouTube
import json
import os
# from datasets import load_dataset, load_from_disk

# dataset = load_dataset("OpenGVLab/InternVid", 'InternVid-10M', cache_dir="datasets/internvid")
# dataset.save_to_disk("datasets/internvid")
jsonl_file_path = 'datasets/internvid/InternVid-10M-flt.jsonl'

with open(jsonl_file_path, 'r') as file:
    for line in file:
        json_obj = json.loads(line)
        video_id = json_obj['YoutubeID']
        aesthetic_score = json_obj['Aesthetic_Score']
        if aesthetic_score < 6.0:
            continue
        else:
            if os.path.exists(os.path.join("datasets/internvid/internVid-10M-flt/original_video/",
                                           f"{video_id}.mp4")):
                print(f"{video_id}.mp4 has been download")
                continue
            video_link = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(video_link)
            try:
                video_stream = yt.streams.get_highest_resolution()
            except:
                continue
            video_stream.download(f"datasets/internvid/internVid-10M-flt/original_video/", filename=f"{video_id}.mp4")
            print('successfully download')
