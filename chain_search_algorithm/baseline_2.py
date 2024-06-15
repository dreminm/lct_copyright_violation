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



def chain_search_algorithm(
    audio: np.array,
    sample_rate: int,
    duration: int,
    window_duration: int = 10,
    window_shift_duration: int = 1,
    distance_threshold: float = 5.0,
    collection_name: str = "audio_segments_ausil",
    chain_shift_duration: int = 1,
    candidates_limit: int = 30,
) -> List[INTERSECTION_OUTPUT]:
    client = MilvusClient(MILVUS_URL)

    result: List[INTERSECTION_OUTPUT] = []

    video_id_to_candidate: Dict[str, Dict[int, List[int]]] = defaultdict(defaultdict(list))

    for start_segment in trange(0, duration, window_shift_duration):
        sub_audio = audio[start_segment * sample_rate : (start_segment + window_duration) * sample_rate]
        embedding = calculate_embedding(sub_audio)

        milvus_output = client.search(
            collection_name=collection_name,
            data=embedding,
            output_fields=["id", "video_id", "segment_start", "embedding"],
            limit=candidates_limit
        )
        if len(milvus_output) == 0:
            continue
        def check(emb):
            return np.array(emb - embedding).__pow__(2).sum().__pow__(0.5) <= distance_threshold
        mulvus_output = milvus_output[0]
        candidates: List[CANDIDATE_TYPE] = [
            (item["entity"]["video_id"], item["entity"]["segment_start"]) for item in milvus_output \
                if check(item["entity"]["embedding"])
        ]
        for video_id, v_seg_start in candidates:
            video_id_to_candidate[video_id][v_seg_start] += [start_segment]

    def add_to_result(video_id: str, v_seg_start: int, seg_starts: Iterable[int], shift: int):
        if shift < MINIMAL_DURATION_VERDICT:
            return
        for seg_start in seg_starts:
            result.append((video_id, v_seg_start, v_seg_start + shift, seg_start, seg_start + shift))

    shift = window_duration
    for video_id, seg_starts_to_candidates in video_id_to_candidate.items():
        keys = sorted(list(seg_starts_to_candidates.keys()))
        max_time = keys[-1]
        for v_seg_start in keys:
            v_init = v_seg_start
            candidates = set(seg_starts_to_candidates[v_init])
            resulted_candidates = set()
            while v_end <= max_time and len(candidates) > 0:
                v_end = v_seg_start + shift
                expected_candidates = set([x + shift for x in candidates])
                real_candidates = set(seg_starts_to_candidates[v_end])
                resulted_candidates = expected_candidates & real_candidates
                to_out_candidates = expected_candidates.difference(real_candidates)
                add_to_result(video_id, v_init, to_out_candidates, v_seg_start - v_init)
                v_seg_start += shift
            add_to_result(video_id, v_init, resulted_candidates, v_seg_start - v_init)

    return result
