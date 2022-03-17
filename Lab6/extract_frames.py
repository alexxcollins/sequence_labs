import os

# this assumes you are in the working directory for Lecture7, and the videos are stored in directory UCF11_updated_mpg 
c_d = os.listdir('.')
if 'data' not in c_d:
    os.mkdir('data')

main_source_dir = 'ucf11_combined'
source_data = sorted(os.listdir(main_source_dir), key = lambda x: x.lower())
target_data = os.listdir('data')
print(source_data)
# this extracts frames from one video for every class
for d in source_data:
    # this returns allvideos in the directory
    all_video_dir = sorted(os.listdir(os.path.join(main_source_dir, d)), key= lambda x:x.lower())
    source_video_file = all_video_dir[0]
    # create the directory for the frames from just this video
    target_dir = os.path.join('data', d+'_0')    
    if not d+'_0' in os.listdir('data'):
       os.mkdir(target_dir)
    # use ffmpeg to extract 1 FPS into the target dir
    os.system("ffmpeg -i {0:} {1:}/%05d.jpg".format(os.path.join(os.path.join(main_source_dir, d), source_video_file), target_dir)) 
