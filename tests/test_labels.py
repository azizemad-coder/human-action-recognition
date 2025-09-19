from src.har_infer.labels import DATASET_ID_TO_LABEL, DATASET_LABEL_TO_ID


def test_labels_roundtrip():
    for name, idx in DATASET_LABEL_TO_ID.items():
        assert DATASET_ID_TO_LABEL[idx] == name


