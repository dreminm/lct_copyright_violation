# import nemo.collections.asr as nemo_asr
import requests
import os
import os.path as osp
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
# from moviepy.editor import VideoFileClip, clips_array
from pathlib import Path
from decord import VideoReader, cpu

from tqdm import tqdm


from baseline_3 import filter_candidates
from glob import glob


if __name__ == "__main__":
    os.environ['BASE_PATH'] = osp.abspath('..')
    os.environ['TEST_PATH'] = osp.abspath('../data/compressed_test')
    os.environ['DATABASE_VIDEOS_PATH'] = osp.abspath('../data/videos')
    files = glob(osp.join(os.environ['TEST_PATH'], '*.mp4'))

    def check(fpath):
        filter = [
            "0v6hb6lkgyfj9i0zra8okhp2vbg6vy7y",
            "4nwuggfnbubduoc0rgtpu0c52jqnc49a"
        ]
        for y in filter:
            if y in fpath:
                return True
        return False

    files = [
        x for x in files if check(x)
    ]

    print(files)
    print(f"Count of test videos: {len(files)}")

    OUTPUT_FOLDER_2 = os.path.join(os.environ['BASE_PATH'], 'output_stage_2')
    OUTPUT_FOLDER_1 = os.path.join(os.environ['BASE_PATH'], 'output_stage_1')

    if not os.path.exists(OUTPUT_FOLDER_2):
        os.makedirs(OUTPUT_FOLDER_2)

    SR = 16_000

    for idx, filepath in enumerate(files):
        basename = Path(filepath).stem
        with open(osp.join(OUTPUT_FOLDER_1, basename + '.json'), 'r') as fin:
            candidates = json.load(fin)

        result = dict()
        source_reader = VideoReader(filepath, ctx=cpu(0))

        for video_id, candidates_list in candidates.items():
            video_check_reader = VideoReader(
                osp.join(
                    os.environ['DATABASE_VIDEOS_PATH'],
                    video_id
                ),
                ctx=cpu(0)
            )
            selected_cand = filter_candidates(
                source_reader, video_check_reader,
                candidates_list
            )
            if len(selected_cand) > 0:
                result[video_id] = selected_cand

        with open(osp.join(OUTPUT_FOLDER_2, basename + '.json'), 'w') as fout:
            json.dump(result, fout, indent=4)

        print(f"Count of processed test videos - {idx + 1}, total count - {len(files)} ")
