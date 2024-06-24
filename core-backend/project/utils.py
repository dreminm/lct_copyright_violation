import requests

import numpy as np

from pymilvus import MilvusClient
from torch import nn, Tensor
from typing import List, Dict, Optional, Tuple, Iterable, Sequence
from tqdm.auto import tqdm, trange


import cv2
import os.path as osp

import numpy as np
import librosa

from decord import VideoReader, cpu
from pathlib import Path

from tqdm.auto import tqdm
from typing import List, Tuple

from collections import defaultdict
from project.settings import _settings


CANDIDATE_TYPE = Tuple[str, int]
INTERSECTION_OUTPUT = Tuple[int, int, int, int]
MINIMAL_DURATION_VERDICT = 10


def calculate_embedding(audio: np.ndarray) -> np.ndarray:
    result = requests.post(
        _settings.embedder_endpoint,
        json={
            "audio": [float(x) for x in audio.tolist()]
        }
    )
    return result.json()["embedding"]


def merge_intervals(intervals):
    if not intervals:
        return []

    # Сортируем интервалы по первому элементу каждого интервала
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] <= last[-1]:  # Проверяем пересечение
            # Объединяем интервалы
            merged[-1] = sorted(list(set(last + current)))
        else:
            merged.append(current)
    merged = [
        {
            "density": len(elem)/(elem[-1][0]-elem[0][0]+1),
            "path": elem
        }
        for elem in merged if len(elem)>10
    ]
    
    return merged


def chain_search_algorithm(
    audio: np.array,
    sample_rate: int,
    duration: int,
    window_duration: int = 10,
    window_shift_duration: int = 1,
    distance_threshold: float = 5.0,
    collection_name: str = "whisper__10__0",
    chain_shift_duration: int = 1,
    candidates_limit: int = 30,
) -> Dict[str, List[int]]:
    client = MilvusClient(_settings.milvus_endpoint)

    result: List[INTERSECTION_OUTPUT] = []

    paths = {}

    for start_segment in trange(0, duration, window_shift_duration):
        sub_audio = audio[start_segment * sample_rate : (start_segment + window_duration) * sample_rate]
        embedding = calculate_embedding(sub_audio)

        milvus_output = client.search(
            collection_name=collection_name,
            data=embedding,
            output_fields=["id", "video_id", "segment_start"],
            params={
                "range_filter" : distance_threshold
            },
            limit=candidates_limit
        )
        if len(milvus_output) == 0:
            continue
        milvus_output = milvus_output[0]
        candidates: List[CANDIDATE_TYPE] = [
            (item["entity"]["video_id"], item["entity"]["segment_start"]) for item in milvus_output
        ]
        search_result = {}
        for video_id, segment_start in candidates:
            if video_id not in search_result:
                search_result[video_id] = []

            search_result[video_id].append(segment_start)

        search_result = {k: sorted(v) for k, v in search_result.items()}
        
        for video_id, candidates in search_result.items():
            if video_id not in paths:
                paths[video_id]  =  []
            for candidate in candidates:
                used = False
                for idx, path in enumerate(paths[video_id]):
                    if candidate - path[-1][0] < 10 and candidate - path[-1][0] > 0 \
                        and start_segment - path[-1][1] < 10 and start_segment - path[-1][1] > 0:
                        paths[video_id][idx].append((candidate, start_segment))
                        used = True
                    elif candidate - path[-1][0] < 0 and start_segment - path[-1][1] < 0:
                        used = True
                      
                if not used:
                    paths[video_id].append([(candidate, start_segment)])

    # fin_paths = {
    #     k: [interval for interval in v if len(interval)>1]
    #     for k, v in paths.items()
    # }

    # fin_paths = {
    #     k: merge_intervals(v) for k, v in fin_paths.items()
    # }

    # fin_paths = {
    #     k: sorted(v, key=lambda x: x["density"], reverse=True)
    #     for k, v in fin_paths.items() if len(v)>0
    # }

    # fin_paths = {
    #     k: [elem for elem in v if elem["density"]>0.8]
    #     for k, v in fin_paths.items()
    # }

    # fin_paths = {
    #     k: v
    #     for k, v in fin_paths.items() if len(v)>0
    # }
    # paths = dict()
    # for v_id, chains in fin_paths.items():
    #     paths[v_id] = [x['path'] for x in chains]

    result = dict()
    for video_id, chains in paths.items():
        temp = []
        for chain in chains:
            source_start = min((x[1] for x in chain))
            source_end = max((x[1] for x in chain))
            video_seg_start = min((x[0] for x in chain))
            video_seg_end = max((x[0] for x in chain))
            if source_end - source_start >= MINIMAL_DURATION_VERDICT and video_seg_end - video_seg_start >= MINIMAL_DURATION_VERDICT:
                shift = min(video_seg_end - video_seg_start, source_end - source_start)
                temp += [(video_seg_start, video_seg_start + shift, source_start, source_start + shift)]
        if len(temp) > 0:
            result[video_id] = temp
    
    return merge_pieces(result)


def is_intersect(left, right, Left, Right):
    if left <= Left and Right <= right:
        return True
    if Left <= left and right <= Right:
        return True
    if right <= Left or Left >= right:
        return False
    intersection_left = max(left, Left)
    intersection_right = min(right, Right)
    union_left = min(left, Left)
    union_right = max(right, Right)
    iou = float(intersection_right - intersection_left) / (union_right - union_left)
    #print(iou)
    return iou > 0.5


def merge_pieces(segments: Dict[str, List[Tuple[int, int, int, int]]]):
    result = dict()
    for video_id, chains in segments.items():
        # if video_id != 'ded3d179001b3f679a0101be95405d2c.mp4':
        #     continue
        prime_indicator = list(range(len(chains)))
        def get_ancestor(id):
            if prime_indicator[id] == id:
                return id
            prime_indicator[id] = get_ancestor(prime_indicator[id])
            return prime_indicator[id]

        for chain_id, chain in enumerate(chains):
            for chain_id_2, chain_2 in enumerate(chains):
                if chain_id_2 == chain_id: # or prime_indicator[chain_id_2] != chain_id_2:
                    continue
                v_s, v_e, s_s, s_e = chain
                v_s_2, v_e_2, s_s_2, s_e_2 = chain_2
                if not is_intersect(v_s, v_e, v_s_2, v_e_2) or not is_intersect(s_s, s_e, s_s_2, s_e_2):
                    continue
                prime_indicator[chain_id_2] = get_ancestor(chain_id)
        distinct_segments = [[] for _ in chains]
        for prime_ind, chain in zip(prime_indicator, chains):
            distinct_segments[prime_ind] += [chain]
        temp = []
        for d_seg in distinct_segments:
            #print(d_seg)
            if len(d_seg) == 0:
                continue
            seg = np.array(d_seg)
            seg = np.array(
                [
                    np.min(seg[:, 0]),
                    np.max(seg[:, 1]),
                    np.min(seg[:, 2]),
                    np.max(seg[:, 3]),
                ]
            )
            seg = [int(np.ceil(elem)) for elem in seg]
            shift = min(seg[1] - seg[0], seg[3] - seg[2])
            if shift >= MINIMAL_DURATION_VERDICT:
                temp += [[seg[0], seg[0] + shift, seg[2], seg[2] + shift]]
        if len(temp) > 0:
            result[video_id] = temp
    return result


FPS = 10
FRAMES_SHIFT = 5
MIN_COUNT = 3


def get_matchings_count(
    keypoints_1,
    descriptions_1,
    keypoints_2,
    descriptions_2,
):
    if len(descriptions_2) == 0 or len(descriptions_1) == 0:
        return 0
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptions_1, descriptions_2,k=2)
    
    # Apply ratio test
    answer = 0
    try:
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                answer += 1
    except:
        pass
    return answer


def calculate_sift_features(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = frame.shape[:2]

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(frame, None)
    if des is None:
        kp = []
        des = []
    kp_res = []
    des_res = []
    for k, d in zip(kp, des):
        x, y = k.pt
        x /= w
        y /= h
        if (35 / 360. <= y <= 50 / 360.) and (550 / 640. <= x <= 620 / 640):

            continue
        else:
            kp_res += [k]
            des_res += [d]
    return kp_res, np.array(des_res)


def filter_candidates(
    source_video: VideoReader,
    check_video_reader: VideoReader,
    candidates: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    result = []
    for v_s, v_e, s_s, s_e in tqdm(candidates):
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
                #print(len(s_kp_des[0]), len(s_kp_des[1]), flush=True)
                #print(len(v_kp_des[0]), len(v_kp_des[1]), flush=True)
                matrix[s_idx, v_idx] = get_matchings_count(*s_kp_des, *v_kp_des)

        best_s = None
        best_v = None
        best_len = 0
        dp = np.zeros_like(matrix).astype(int)
        for s_idx in range(0, len(sift_kp_des_s)):
            dp[s_idx][0] = int(matrix[s_idx][0] >= MIN_COUNT)
        for v_idx in range(0, len(sift_kp_des_v)):
            dp[0][v_idx] = int(matrix[0][v_idx] >= MIN_COUNT)
        for s_idx in range(1, len(sift_kp_des_s)):
            for v_idx in range(1, len(sift_kp_des_v)):
                if matrix[s_idx, v_idx] >= MIN_COUNT:
                    dp[s_idx][v_idx] = dp[s_idx - 1][v_idx - 1] + 1

        for s_idx in range(0, len(sift_kp_des_s)):
            for v_idx in range(0, len(sift_kp_des_v)):
                cur_len = dp[s_idx][v_idx]
                if best_len < cur_len:
                    best_len = int(cur_len)
                    best_s = int(s_secs[s_idx - cur_len + 1] // FPS)
                    best_v = int(v_secs[v_idx - cur_len + 1] // FPS)
        if best_len >= 10:
            result += [(best_v, best_v + best_len, best_s, best_s + best_len)]

    return result


def inference_algorithm(file_path: str, database_videos: str, SR: int = 16_000, collection_name: str = "whisper_segments"):
    audio, _ = librosa.load(file_path, sr=SR)
    duration = len(audio) // SR
    audio_candidates = chain_search_algorithm(
        audio,
        SR,
        duration,
        10,
        1,
        2,
        collection_name,
        None,
        100
    )
    result = dict()
    source_reader = VideoReader(file_path, ctx=cpu(0))

    candidates = audio_candidates

    for video_id, candidates_list in candidates.items():
        video_check_reader = VideoReader(
            osp.join(
                database_videos, #os.environ['DATABASE_VIDEOS_PATH'],
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
    return format_result(result)


def second_to_string(second: int):
    sec = second % 60
    minutes = second // 60
    mins = minutes % 60
    hours = minutes // 60
    assert hours < 24
    return "{:02d}:{:02d}:{:02d}".format(hours, mins, sec)


def format_result(result: Dict[str, List[Tuple[int, int, int, int]]]):
    #     analog_info = [
    #             {"filename": "http://localhost:2001/api/video/video2.mp4", "time_intervals": [
    #                 {"start_sec": 27, "t_start": "0:0:27 - 0:0:50"},
    #                 {"start_sec": 30, "t_start": "0:0:30 - 0:0:53"}]},
    #             {"filename": "http://localhost:2001/api/video/video3.mp4", "time_intervals": [
    #                 {"start_sec": 125, "t_start": "0:2:5 - 0:2:20"}]}
    #         ]
    #         upload_info = [
    #             [{"start_sec": 40, "t_start": "0:0:40 - 0:1:3"},
    #              {"start_sec": 45, "t_start": "0:0:45 - 0:1:8"}],
    #             [{"start_sec": 140, "t_start": "0:2:20 - 0:2:35"}]
    #         ]
    analog_info = []
    upload_info = []
    for video_id, candidates_list in result.items():
        time_intervals_analog = []
        time_intervals_upload = []
        for v_s, v_e, s_s, s_e in candidates_list:
            time_intervals_analog += [
                {
                    "start_sec": v_s,
                    "t_start": second_to_string(v_s) + " - " + second_to_string(v_e)
                }
            ]
            time_intervals_upload += [
                {
                    "start_sec": s_s,
                    "t_start": second_to_string(s_s) + " - " + second_to_string(s_e)
                }
            ]
        analog_info += [
            {"filename": f"http://localhost:12345/files/{video_id}", "time_intervals": time_intervals_analog}
        ]
        upload_info.append(time_intervals_upload)
    return analog_info, upload_info
