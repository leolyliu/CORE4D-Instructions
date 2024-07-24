import os


def avi2depth(video_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.system("ffmpeg -i {} -f image2 -start_number 0 -vf fps=fps=15 -qscale:v 2 {}/%d.png -loglevel quiet".format(video_path, save_dir))
