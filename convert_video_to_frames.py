
import av
import tqdm
import os
import multiprocessing as mp
import argparse





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='/home/kouyou/datasets/krishnacam/videos', help='Root directory of video dataset.')
    parser.add_argument('--frames_dir', default='/home/kouyou/datasets/krishnacam/frames', help='Root directory to place frames.')
    parser.add_argument('--workers', default=0, type=int, help='Number of parallel workers.')
    args = parser.parse_args()
    

    videos_fns = [f"{root}/{fn}" for root, subdir, files in os.walk(args.video_dir) for fn in files if fn.endswith('.mp4') or fn.endswith('.avi')]


    # 30 fpsで作成

    jobs = []
    for vfn in videos_fns:
        basename = os.path.basename(vfn).split('.')[0]
        dst_fns = f'{args.frames_dir}/{basename}/%07d.jpg'
        os.makedirs(os.path.dirname(dst_fns), exist_ok=True)




if __name__ == "__main__":
    main()





