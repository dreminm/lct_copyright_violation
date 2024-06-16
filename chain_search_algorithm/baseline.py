import torch
import requests

import numpy as np

from pymilvus import MilvusClient
from torch import nn, Tensor
from typing import List, Dict, Optional, Tuple
from tqdm.auto import tqdm, trange

from constants import MILVUS_URL, EMBEDDER_URL, MINIMAL_DURATION_VERDICT


CANDIDATE_TYPE = Tuple[str, str, int]
INTERSECTION_OUTPUT = Tuple[str, int, int, int, int]


def calculate_embedding(audio: np.ndarray) -> np.ndarray:
    result = requests.post(
        EMBEDDER_URL,
        json={
            "audio": [float(x) for x in audio.tolist()]
        }
    )
    return result.json()["embedding"]


def select_consistent_chains(
    candidates: List[CANDIDATE_TYPE],
    audio: np.ndarray,
    sample_rate: int,
    duration: int,
    start_segment: int,
    embedding: np.ndarray,
    distance_threshold: float,
    chain_shift_duration: int,
    start_window_duration: int,
    milvus_client: MilvusClient,
    collection_name: str
) -> Tuple[List[INTERSECTION_OUTPUT], int]:
    result: List[INTERSECTION_OUTPUT] = []

    for shift in [0] + list(range(start_window_duration, duration - start_segment, chain_shift_duration)):
        if len(candidates) == 0:
            break

        if shift > 0:
            embedding = calculate_embedding(
                audio[
                    (start_segment + shift) * sample_rate : (start_segment + shift + chain_shift_duration) * sample_rate
                ]
            )
        candidates_not_do_delete = []

        for (id, video_id, seg_start) in candidates:
            cand_embedding = milvus_client.query(
                collection_name=collection_name,
                filter = f"video_id == '{video_id}' and segment_start == {seg_start + shift}"
            )
            if len(cand_embedding) == 0:
                continue
            cand_embedding = cand_embedding[0]["embedding"]
            cand_embedding = np.array(cand_embedding)
            diff = np.array(cand_embedding - embedding)
            if diff.__pow__(2).sum().__pow__(0.5) <= distance_threshold:
                candidates_not_do_delete += [(id, video_id, seg_start)]
            elif shift - chain_shift_duration >= MINIMAL_DURATION_VERDICT:
                episode_duration = shift - chain_shift_duration
                result += [
                    (video_id, start_segment, start_segment + episode_duration, seg_start, seg_start + episode_duration)
                ]

        candidates = candidates_not_do_delete

    if len(candidates) > 0:
        for (id, video_id, seg_start) in candidates:
            episode_duration = shift
            result += [
                (video_id, start_segment, start_segment + episode_duration, seg_start, seg_start + episode_duration)
            ] 

    return result, shift


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

    # start_segment = 0
    # while start_segment < duration:
    shift = 0
    for start_segment in trange(0, duration, window_shift_duration):
        #start_segment = max(start_segment, shift)
        sub_audio = audio[start_segment * sample_rate : (start_segment + window_duration) * sample_rate]
        embedding = calculate_embedding(sub_audio)

        milvus_output = client.search(
            collection_name=collection_name,
            data=embedding,
            output_fields=["id", "video_id", "segment_start"],
            limit=candidates_limit
        )[0]
        candidates: List[CANDIDATE_TYPE] = [
            (item["entity"]["id"], item["entity"]["video_id"], item["entity"]["segment_start"]) for item in milvus_output
        ]
        output, shift = select_consistent_chains(
            candidates,
            audio, sample_rate, duration, start_segment,
            embedding, distance_threshold, chain_shift_duration, window_duration,
            client, collection_name
        )
        result += output
        #shift = start_segment + shift
        #start_segment += window_shift_duration # max(shift, window_shift_duration)

    return select_not_intersected_segments(result)


def select_not_intersected_segments(result: List[INTERSECTION_OUTPUT]) -> List[INTERSECTION_OUTPUT]:
    return result
    # selected = []
    # intersected_with = [None] * len(result)
    # for (video_id, s)
    # return selected
