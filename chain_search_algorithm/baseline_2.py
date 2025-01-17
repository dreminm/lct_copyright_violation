import torch
import requests

import numpy as np

from pymilvus import MilvusClient
from torch import nn, Tensor
from typing import List, Dict, Optional, Tuple, Iterable, Sequence
from tqdm.auto import tqdm, trange

from collections import defaultdict
from constants import MILVUS_URL, EMBEDDER_URL, MINIMAL_DURATION_VERDICT


CANDIDATE_TYPE = Tuple[str, int]
INTERSECTION_OUTPUT = Tuple[str, int, int, int, int]


def calculate_embedding(audio: np.ndarray) -> np.ndarray:
    result = requests.post(
        EMBEDDER_URL,
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
    client = MilvusClient(MILVUS_URL)

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
