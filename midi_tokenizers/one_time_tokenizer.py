from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer


class OneTimeTokenizer(NoLossTokenizer):
    def __init__(
        self,
        min_time_unit: float = 0.001,
        n_velocity_bins: int = 128,
    ):
        super().__init__(min_time_unit=min_time_unit, n_velocity_bins=n_velocity_bins)
        self.name = "OneTimeTokenizer"

    def _time_vocab(self):
        return ["1T"], {"1T": self.min_time_unit}, {self.min_time_unit: "1T"}
