import os
from pathlib import Path

main_source_dir = Path('../data/lab6/ucf11_combined')
# this sorts all files except files starting with '.' - e.g. .ipynb_checkpoints
source_data = sorted(main_source_dir.glob('[!.]*'))
target_data = Path('../data/lab6')

##### from ffmpeg docs:
# ffmpeg -i movie.mpg movie%d.jpg
# https://ffmpeg.org/faq.html#How-do-I-encode-movie-to-single-pictures_003f
#####

train = []
test = []
val = []
for d in source_data:
    # ignore .ipynb_checkpoint folder. Don't need to sort in a special way.
    all_vids = sorted(d.glob('[!.]*'))
    # videos are grouped into 25 groups with similarities within group.
    # so groups cannot be split across test/train/split. File name is of format
    # v_[category]_[group]_[numnber within group], so split by group number
    # into test/train/split
    for vid in all_vids:
        group = int(vid.stem.split('_')[-2])
        if group < 16:
            train.append(vid)
        elif group >15 and group <21:
            test.append(vid)
        elif group >20:
            val.append(vid)

# iterate through test/train/split, split vids into frames and save in
# test/train/val folder. Eech video has its own directory.
for test_set, set_str in ((train, 'train'), (test, 'test'), (val, 'val')):
    for vid in test_set:
        p = target_data/('ucf11_' + set_str + '_data')
        p = p/vid.stem
        p.mkdir(exist_ok=True)
        os.system("ffmpeg -i {:} {:}/%05d.jpg".format(str(vid), str(p)))