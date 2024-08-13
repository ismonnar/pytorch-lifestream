from typing import List, Dict

from torch.utils.data import DataLoader

from ptls.data_load import IterableChain
from ptls.data_load.utils import collate_feature_for_inference
from ptls.data_load.filter_dataset import FilterDataset
from ptls.data_load.iterable_processing import ToTorch, FilterNonArray, ISeqLenLimit


def inference_data_loader(
    data: List[Dict],
    max_seq_len: int = 10000,
    num_workers: int = 0,
    batch_size: int = 512,
    onnx = False
):
    r"""Generate an inference data loader

    Parameters
    ----------
    data: List[Dict]
        The dataset
    max_seq_len:
        Only `max_seq_len` transaction will taken from tail of sequence.
        Unlimited sequence length lead to out of memory error
    num_workers: int. Default: 0.
        The number of workers for the dataloader. 0 = single-process loader
    batch_size: int. Default: 512.
        The number of samples (before splitting to subsequences) in each batch
    onnx: bool. Default: False
        Flag for ONNX export
    """
    dataset = FilterDataset(
        data,
        post_processing=IterableChain(
            ToTorch(),
            FilterNonArray(),
            # ISeqLenLimit(max_seq_len=max_seq_len),
        )
    )

    return DataLoader(
        dataset=dataset,
        collate_fn=collate_feature_for_inference,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
    )
