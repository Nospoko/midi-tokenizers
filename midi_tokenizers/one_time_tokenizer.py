from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer


class OneTimeTokenizer(ExponentialTimeTokenizer):
    """
    A tokenizer that uses a single time unit for quantization.

    Inherits from ExponentialTimeTokenizer and overrides the _time_vocab method
    to use a fixed time unit.

    Attributes:
        min_time_unit (float): The minimum time unit for quantizing time.
        n_velocity_bins (int): The number of velocity bins.
        special_tokens (list[str]): A list of special tokens.
        name (str): Name of the tokenizer.
    """

    def __init__(
        self,
        min_time_unit: float = 0.001,
        n_velocity_bins: int = 128,
        special_tokens: list[str] = None,
    ):
        """
        Initialize the OneTimeTokenizer with a fixed time unit.

        Parameters:
        min_time_unit (float): The minimum time unit for quantizing time. Defaults to 0.001.
        n_velocity_bins (int): The number of velocity bins. Defaults to 128.
        special_tokens (list[str]): A list of special tokens. Defaults to None.
        """
        super().__init__(
            min_time_unit=min_time_unit,
            n_velocity_bins=n_velocity_bins,
            special_tokens=special_tokens,
        )
        self.name = "OneTimeTokenizer"
        self.pad_token_id = self.token_to_id["<PAD>"]

    def _time_vocab(self):
        return ["1T"], {"1T": self.min_time_unit}, {self.min_time_unit: "1T"}
