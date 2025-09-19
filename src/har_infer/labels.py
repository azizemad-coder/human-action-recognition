from __future__ import annotations

from typing import Dict, List


# Mapping given by the dataset card
DATASET_LABEL_TO_ID: Dict[str, int] = {
    "calling": 0,
    "clapping": 1,
    "cycling": 2,
    "dancing": 3,
    "drinking": 4,
    "eating": 5,
    "fighting": 6,
    "hugging": 7,
    "laughing": 8,
    "listening_to_music": 9,
    "running": 10,
    "sitting": 11,
    "sleeping": 12,
    "texting": 13,
    "using_laptop": 14,
}

DATASET_ID_TO_LABEL: List[str] = [
    label for label, _ in sorted(DATASET_LABEL_TO_ID.items(), key=lambda kv: kv[1])
]


