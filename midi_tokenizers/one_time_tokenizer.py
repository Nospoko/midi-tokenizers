from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer


class OneTimeTokenizer(NoLossTokenizer):
    def __init__(
        self,
        eps: float = 0.001,
        n_velocity_bins: int = 128,
    ):
        super().__init__(eps=eps, n_velocity_bins=n_velocity_bins)
        self.name = "OneTimeTokenizer"

    def _time_vocab(self):
        return ["1T"], {"1T": self.eps}, {self.eps: "1T"}
