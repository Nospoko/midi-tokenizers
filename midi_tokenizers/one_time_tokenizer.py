from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer


class OneTimeTokenizer(NoLossTokenizer):
    def __init__(
        self,
        eps: float = 0.001,
        n_velocity_bins: int = 128,
    ):
        super().__init__(eps=eps, n_velocity_bins=n_velocity_bins)

    def _time_vocab(self):
        return [f"{self.eps}s"]
