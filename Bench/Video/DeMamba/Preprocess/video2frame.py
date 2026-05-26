import os
from glob import glob
from moviepy.editor import VideoFileClip
import multiprocessing
import cv2, math

def get_video_length(file_path):
    video = VideoFileClip(file_path)
    return video.duration

def process_video(video_path):
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    video_name = video_name.split('-')[1:]
    video_name = '-'.join(video_name)

    path = video_path.split('/')[1:-1]
    path = '/'.join(path)

    #image_path = path+'/'+ 'MSRVTT_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'K400_frames/'+ video_name+'/'
    # image_path = path+'/'+ 'SEINE_frames/'+ video_name+'/'
    # image_path = path+'/'+ 'Crafter_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'Gen2_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'HotShot_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'Lavie_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'ModelScope_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'MoonValley_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'MorphStudio_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'Show_1_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'Sora_frames/'+ video_name+'/'
    #image_path = path+'/'+ 'WildScrape_frames/'+ video_name+'/'
    # image_path = path+'/'+ 'AVdataset_frames/'+ video_name+'/'
    image_path = path + '/' + 't2vz_frames/' + video_name + '/'


    print(video_name)
    if os.path.exists(image_path):
        print("路径存在")
    else:
        print(video_name, "路径不存在")
        try:
            try:
                video_length = get_video_length(video_path)
                print(video_name, f"视频长度为：{video_length} 秒")
                os.makedirs(os.path.dirname(image_path), exist_ok=True)

                if video_length >= 4 :
                    inter_val = 2
                    os.system(f"cd {image_path} | ffmpeg -loglevel quiet -i {video_path} -r {inter_val} {image_path}%d.jpg")
                else:
                    inter_val = math.ceil(8 / video_length)
                    os.system(f"cd {image_path} | ffmpeg -loglevel quiet -i {video_path} -r {inter_val} {image_path}%d.jpg")
 
            except Exception as e:
                print("发生异常：", str(e))
        except:
            print("Skip")

if __name__ == '__main__':
    print("Getting frames!!")
    #video_paths = 'GenVideo/real/K400'
    #video_paths = 'GenVideo/real/MSRVTT'
    # video_paths = 'GenVideo/fake/SEINE'
    #video_paths = 'GenVideo/fake/Crafter'
    #video_paths = 'GenVideo/fake/Gen2'
    #video_paths = 'GenVideo/fake/HotShot'
    #video_paths = 'GenVideo/fake/Lavie'
    #video_paths = 'GenVideo/fake/ModelScope'
    #video_paths = 'GenVideo/fake/MoonValley'
    #video_paths = 'GenVideo/fake/MorphStudio'
    #video_paths = 'GenVideo/fake/Show_1'
    #video_paths = 'GenVideo/fake/Sora'
    #video_paths = 'GenVideo/fake/WildScrape'
    # video_paths = 'GenVideo/fake/AVdataset'
    video_paths = 'GenVideo/fake/t2vz'


    
    all_dirs = []
    all_dirs = glob(video_paths+'/*')
    
    
    pool = multiprocessing.Pool(processes=8)
    pool.map(process_video, all_dirs)
    pool.close()
    pool.join()



