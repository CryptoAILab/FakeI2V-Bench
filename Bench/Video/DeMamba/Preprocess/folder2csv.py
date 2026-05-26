import os
import csv
import pandas as pd
from pandas import Series, DataFrame
from glob import glob
import os

def count_images_in_folder(folder_path):
    image_count = 0
    image_names = []  
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            image_count += 1
            image_names.append(int(file_name.split('.')[0]))
    image_names.sort()
    return image_count, image_names


#folder_path = 'real/K400/K400_frames'
#folder_path = 'real/MSRVTT/MSRVTT_frames'
#folder_path = 'fake/SEINE/SEINE_frames'
#folder_path = 'fake/Crafter/Crafter_frames'
#folder_path = 'fake/Gen2/Gen2_frames'
#folder_path = 'fake/HotShot/HotShot_frames'
#folder_path = 'fake/Lavie/Lavie_frames'
#folder_path = 'fake/ModelScope/ModelScope_frames'
#folder_path = 'fake/MoonValley/MoonValley_frames'
#folder_path = 'fake/MorphStudio/MorphStudio_frames'
#folder_path = 'fake/Show_1/Show_1_frames'
#folder_path = 'fake/Sora/Sora_frames'
#folder_path = 'fake/WildScrape/WildScrape_frames'
# folder_path = 'fake/AVdataset/AVdataset_frames'
# folder_path = 'fake/K400Rec/full_inpainting'
folder_path = 'fake/t2vz/t2vz_frames'

all_dirs = []

for root, dirs, files in os.walk(folder_path):
    for dir in dirs:
        all_dirs.append(os.path.join(root, dir))

label = list()
save_path = list()
frame_counts = list()
frame_seq_counts = list()
content_paths = list()
chinese_labels = list()


for video_path in all_dirs:
    frame_paths = glob(video_path + '/*')
    temp_frame_count, temp_frame_seqs = count_images_in_folder(video_path)
    if temp_frame_count == 0:
        continue

    for frame in frame_paths:
        content_path = frame.split('/')[:-1]
        content_path = '/'.join(content_path)
        # input your own path
        #content_path = '/MSRVTT/' + content_path
      
        #frame_path = frame.split('/')[1:]
        frame_path = frame.split('/')
        frame_path = '/'.join(frame_path)
        #frame_path = '/MSRVTT/' + frame_path

        print(content_path, frame_path)
        #label.append(str(0)) #real
        label.append(str(1)) #fake
        frame_counts.append(int(temp_frame_count))
        frame_seq_counts.append(temp_frame_seqs)
        save_path.append(frame_path)
        content_paths.append(content_path)
        #chinese_labels.append('真实视频')
        chinese_labels.append('AIGC视频')
        #break

dic={
    'content_path': Series(data=content_paths),
    'image_path': Series(data=save_path),
    'type_id': Series(data=chinese_labels),
    'label': Series(data=label),
    'frame_len': Series(data=frame_counts),
    'frame_seq': Series(data=frame_seq_counts)
}

print(dic)
#pd.DataFrame(dic).to_csv('MSRVTT.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('K400.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('SEINE.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('Crafter.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('Gen2.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('HotShot.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('Lavie.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('ModelScope.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('MoonValley.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('MorphStudio.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('Show_1.csv', encoding='utf-8', index=False)
#pd.DataFrame(dic).to_csv('Sora.csv', encoding='utf-8', index=False)
# pd.DataFrame(dic).to_csv('AVdataset.csv', encoding='utf-8', index=False)
# pd.DataFrame(dic).to_csv('K400Rec.csv', encoding='utf-8', index=False)
pd.DataFrame(dic).to_csv('t2vz.csv', encoding = 'utf-8', index = False)

