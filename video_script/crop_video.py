from moviepy.video.io.VideoFileClip import VideoFileClip
import json
import os

jsonl_file_path = 'datasets/internvid/InternVid-10M-flt.jsonl'
output_path = 'datasets/internvid/internVid-10M-flt/crop_video'
deal_sum = 0

def trim_video(input_path, output_path, start_time, end_time):
    # load video
    clip = VideoFileClip(input_path)
    # crop video
    trimmed_clip = clip.subclip(start_time, end_time)
    # save video
    trimmed_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

if os.path.exists(os.path.join(output_path, 'video_txt.json')):
    with open(os.path.join(output_path, 'video_txt.json'), 'r') as f:
        out_dict = json.load(f)
else:
    out_dict = {}

# calculate the video id
id = len(out_dict.keys())

with open(jsonl_file_path, 'r') as file:
    for line in file:
        json_obj = json.loads(line)
        video_id = json_obj['YoutubeID']
        aesthetic_score = json_obj['Aesthetic_Score']
        description = json_obj['Caption']
        start_time = json_obj['Start_timestamp']
        end_time = json_obj['End_timestamp']
        if aesthetic_score < 6.0:
            continue
        else:
            input_video_path = f"datasets/internvid/internVid-10M-flt/original_video/{video_id}.mp4"
            if os.path.exists(input_video_path) == False:
                continue
            output_file_name = f"{video_id}_{start_time}_{end_time}.mp4"
            if os.path.exists(os.path.join(output_path, output_file_name)):
                continue
            out_dict[output_file_name] = {"text": description, "id": id}
            try:
                trim_video(input_video_path,
                        output_path=os.path.join(output_path, output_file_name),
                        start_time=start_time,
                        end_time=end_time)
            except:
                continue
            with open(os.path.join(output_path, 'video_txt.json'), 'w') as file:
                json.dump(out_dict, file, indent=4)
            deal_sum += 1
        
        if deal_sum >= 10:
            break
        id += 1
