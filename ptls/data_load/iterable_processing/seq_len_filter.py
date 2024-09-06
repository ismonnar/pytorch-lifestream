from ptls.data_load import IterableProcessingDataset


class SeqLenFilter(IterableProcessingDataset):
    """
    Filter sequences by length. Drop sequences shorter than `min_seq_len` and longer than `max_seq_len`.


    Args:
        min_seq_len: if set than drop sequences shorter than `min_seq_len`
        max_seq_len: if set than drop sequences longer than `max_seq_len`
        seq_len_col: field where sequence length stored, if None, `target_col` used
        sequence_col: field for sequence length detection, if None, any iterable field will be used
    """
    def __init__(self,
                 min_seq_len: int = None,
                 max_seq_len: int = None,
                 seq_len_col: int = None,
                 sequence_col=None
                 ):
        super().__init__()

        self._min_seq_len = min_seq_len
        self._max_seq_len = max_seq_len
        self._sequence_col = sequence_col
        self._seq_len_col = seq_len_col

    def __iter__(self):
        for rec in self._src:
            features = rec[0] if type(rec) is tuple else rec
            seq_len = self.get_len(features)
            if self._min_seq_len is not None and seq_len < self._min_seq_len:
                continue
            if self._max_seq_len is not None and seq_len > self._max_seq_len:
                continue
            yield rec

    def get_len(self, rec):
        if self._seq_len_col is not None:
            return rec[self._seq_len_col]
        return len(rec[self.get_sequence_col(rec)])
