import cv2

import numpy as np

from decord import VideoReader, cpu
from pathlib import Path

from typing import List, Tuple


FPS = 10
FRAMES_SHIFT = 5
MIN_COUNT = 3


def get_matchings_count(
    keypoints_1,
    descriptions_1,
    keypoints_2,
    descriptions_2,
):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptions_1, descriptions_2,k=2)
    
    # Apply ratio test
    answer = 0
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            answer += 1
    return answer


def calculate_sift_features(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(frame, None)
    return kp, des


def filter_candidates(
    source_video: VideoReader,
    check_video_reader: VideoReader,
    candidates: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    result = []
    for v_s, v_e, s_s, s_e in candidates:
        sift_kp_des_v = []
        sift_kp_des_s = []
        secs_arrays = []
        for (start, end, dst_array, video_reader) in [
            (v_s, v_e, sift_kp_des_v, check_video_reader),
            (s_s, s_e, sift_kp_des_s, source_video)
        ]:
            frames_count = len(video_reader)
            duration = frames_count // FPS
            secs = [x * FPS for x in range(max(0, start - FRAMES_SHIFT), min(duration, end + FRAMES_SHIFT))]
            frames = video_reader.get_batch(secs).asnumpy()
            for frame in frames:
                dst_array += [calculate_sift_features(frame)]
            secs_arrays += [secs]
        v_secs = secs_arrays[0]
        s_secs = secs_arrays[1]
        matrix = np.zeros((len(s_secs), len(v_secs)))
        for s_idx, s_kp_des in enumerate(sift_kp_des_s):
            for v_idx, v_kp_des in enumerate(sift_kp_des_v):
                matrix[s_idx, v_idx] = get_matchings_count(*s_kp_des, *v_kp_des)

        best_s = None
        best_v = None
        best_len = 0
        for s_idx in range(len(sift_kp_des_s)):
            for v_idx in range(len(sift_kp_des_v)):
                cur_len = 0
                while s_idx + cur_len < len(sift_kp_des_s) and v_idx + cur_len < len(sift_kp_des_v) \
                    and matrix[s_idx + cur_len, v_idx + cur_len] >= MIN_COUNT:
                    cur_len += 1
                if best_len < cur_len:
                    best_len = cur_len
                    best_s = s_idx
                    best_v = v_idx
        if best_len >= 10:
            result += [(best_v, best_v + best_len, best_s, best_s + best_len)]

    return result
