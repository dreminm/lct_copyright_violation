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

from joblib import Parallel, delayed
from tqdm import tqdm


from baseline_3 import filter_candidates
from glob import glob


if __name__ == "__main__":
    os.environ['BASE_PATH'] = osp.abspath('..')
    os.environ['TEST_PATH'] = osp.abspath('../data/compressed_test')
    os.environ['DATABASE_VIDEOS_PATH'] = osp.abspath('../data/videos')
    files = glob(osp.join(os.environ['TEST_PATH'], '*.mp4'))

    def weight(fpath):
        stage_1_json = Path(
            osp.join(
                os.path.join(os.environ['BASE_PATH'], 'output_stage_1'),
                Path(fpath).stem + '.json'
            )
        )
        with open(stage_1_json, 'r') as fin:
            js = json.load(fin)
        return -len(js)

    files = sorted(files, key=weight)

    print(files)
    print(f"Count of test videos: {len(files)}")

    OUTPUT_FOLDER_2 = os.path.join(os.environ['BASE_PATH'], 'output_stage_2')
    OUTPUT_FOLDER_1 = os.path.join(os.environ['BASE_PATH'], 'output_stage_1')

    if not os.path.exists(OUTPUT_FOLDER_2):
        os.makedirs(OUTPUT_FOLDER_2)

    SR = 16_000

    def process(selected_f: str):
        basename = Path(selected_f).stem
        stage_2_json_fpath = osp.join(OUTPUT_FOLDER_2, basename + '.json')
        if os.path.exists(stage_2_json_fpath):
            return

        #candidates = {k: v for k, v in candidates.items() if k == "cc0904d3de995d4851de65b93860d8d5.mp4"}
        result = dict()
        source_reader = VideoReader(selected_f, ctx=cpu(0))

        stage_1_json = Path(
            osp.join(
                os.path.join(os.environ['BASE_PATH'], 'output_stage_1'),
                Path(selected_f).stem + '.json'
            )
        )
        with open(stage_1_json, 'r') as fin:
            candidates = json.load(fin)

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

    for f in files:
        process(f)