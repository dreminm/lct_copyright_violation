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

from tqdm import tqdm


from baseline_2 import chain_search_algorithm
from glob import glob


if __name__ == "__main__":
    os.environ['BASE_PATH'] = osp.abspath('..')
    os.environ['TEST_PATH'] = osp.abspath('../data/compressed_test')
    files = glob(osp.join(os.environ['TEST_PATH'], '*.mp4'))
    print(files)
    print(f"Count of test videos: {len(files)}")

    OUTPUT_FOLDER = os.path.join(os.environ['BASE_PATH'], 'output_stage_1')
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    SR = 16_000

    for idx, filepath in enumerate(files):
        basename = Path(filepath).stem
        audio, _ = librosa.load(filepath, sr=SR)
        duration = len(audio) // SR 
        result = chain_search_algorithm(
            audio,
            16_000,
            duration,
            10,
            1,
            2,
            "whisper__10__0__old",
            None,
            100
        )
        with open(osp.join(OUTPUT_FOLDER, basename + '.json'), 'w') as fout:
            json.dump(result, fout, indent=4)

        print(f"Count of processed test videos - {idx + 1}, total count - {len(files)} ")
