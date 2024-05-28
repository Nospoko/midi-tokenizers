import yaml
import numpy as np
import pandas as pd

from midi_quantizers.quantizer import MidiQuantizer


class RelativeTimeQuantizer(MidiQuantizer):
    """
    Quantizer for MIDI data using bins for dstart, duration, and velocity.

    Attributes:
        n_dstart_bins (int): Number of bins for dstart (time between note starts).
        n_duration_bins (int): Number of bins for note durations.
        n_velocity_bins (int): Number of bins for note velocities.
        keys (list[str]): List of quantization keys.
        dstart_bin_edges (np.ndarray): Edges for dstart bins.
        duration_bin_edges (np.ndarray): Edges for duration bins.
        velocity_bin_edges (np.ndarray): Edges for velocity bins.
        bin_to_dstart (list[float]): Mapping from bins to dstart values.
        bin_to_duration (list[float]): Mapping from bins to duration values.
        bin_to_velocity (list[int]): Mapping from bins to velocity values.
    """

    def __init__(
        self,
        n_dstart_bins: int = 3,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
    ):
        """
        Initialize the RelativeTimeQuantizer with specified bins for dstart, duration, and velocity.

        Parameters:
        n_dstart_bins (int): Number of bins for dstart (time between note starts). Defaults to 3.
        n_duration_bins (int): Number of bins for note durations. Defaults to 3.
        n_velocity_bins (int): Number of bins for note velocities. Defaults to 3.
        """
        self.keys = ["pitch", "dstart_bin", "duration_bin", "velocity_bin"]
        self.n_dstart_bins = n_dstart_bins
        self.n_duration_bins = n_duration_bins
        self.n_velocity_bins = n_velocity_bins
        self._build()

    def __rich_repr__(self):
        yield "RelativeTimeQuantizer"
        yield "n_dstart_bins", self.n_dstart_bins
        yield "n_duration_bins", self.n_duration_bins
        yield "n_velocity_bins", self.n_velocity_bins

    def _build(self):
        """
        Build the quantizer by loading bin edges and decoders for dstart, duration, and velocity.
        """
        self._load_bin_edges()
        self._build_dstart_decoder()
        self._build_duration_decoder()
        self._build_velocity_decoder()

    def _load_bin_edges(self):
        """
        Load bin edges for dstart, duration, and velocity from a YAML file.
        """
        artifacts_path = "midi_quantization_artifacts/bin_edges.yaml"
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        self.dstart_bin_edges = bin_edges["dstart"][self.n_dstart_bins]
        self.duration_bin_edges = bin_edges["duration"][self.n_duration_bins]
        self.velocity_bin_edges = bin_edges["velocity"][self.n_velocity_bins]

    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantize the dstart, duration, and velocity values in the DataFrame into bins.

        Parameters:
        df (pd.DataFrame): The DataFrame containing MIDI data.

        Returns:
        pd.DataFrame: The quantized DataFrame.
        """
        next_start = df.start.shift(-1)
        dstart = next_start - df.start
        df["dstart_bin"] = np.digitize(dstart.fillna(0), self.dstart_bin_edges) - 1
        df["duration_bin"] = np.digitize(df.duration, self.duration_bin_edges) - 1
        df["velocity_bin"] = np.digitize(df.velocity, self.velocity_bin_edges) - 1

        return df

    def quantize_velocity(self, velocity: np.array) -> np.array:
        """
        Quantize an array of velocity values into bins.

        Parameters:
        velocity (np.array): Array of velocity values to quantize.

        Returns:
        np.array: Array of quantized velocity values.
        """
        velocity_bins = np.digitize(velocity, self.velocity_bin_edges) - 1
        quantized_velocity = np.array([self.bin_to_velocity[v_bin] for v_bin in velocity_bins])
        return quantized_velocity

    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the quantization to the DataFrame,
        converting bin indices to actual dstart, duration, and velocity values.

        Parameters:
        df (pd.DataFrame): The DataFrame to apply quantization to.

        Returns:
        pd.DataFrame: The DataFrame with quantization applied.
        """
        quant_dstart = df.dstart_bin.map(lambda it: self.bin_to_dstart[it])
        quant_duration = df.duration_bin.map(lambda it: self.bin_to_duration[it])
        df["start"] = quant_dstart.cumsum().shift(1).fillna(0)
        df["end"] = df.start + quant_duration
        df["duration"] = quant_duration
        df["velocity"] = df.velocity_bin.map(lambda it: self.bin_to_velocity[it])
        return df

    def _build_duration_decoder(self):
        """
        Build a decoder to convert duration bins back to duration values.
        """
        self.bin_to_duration = []
        for it in range(1, len(self.duration_bin_edges)):
            duration = (self.duration_bin_edges[it - 1] + self.duration_bin_edges[it]) / 2
            self.bin_to_duration.append(duration)

        last_duration = 2 * self.duration_bin_edges[-1]
        self.bin_to_duration.append(last_duration)

    def _build_dstart_decoder(self):
        """
        Build a decoder to convert dstart bins back to dstart values.
        """
        self.bin_to_dstart = []
        for it in range(1, len(self.dstart_bin_edges)):
            dstart = (self.dstart_bin_edges[it - 1] + self.dstart_bin_edges[it]) / 2
            self.bin_to_dstart.append(dstart)

        last_dstart = 2 * self.dstart_bin_edges[-1]
        self.bin_to_dstart.append(last_dstart)

    def _build_velocity_decoder(self):
        """
        Build a decoder to convert velocity bins back to velocity values.
        """
        # For velocity the first bin is not going to be
        # evenly populated, skewing towards to higher values
        # (who plays with velocity 0?)
        self.bin_to_velocity = [int(0.8 * self.velocity_bin_edges[1])]

        for it in range(2, len(self.velocity_bin_edges)):
            velocity = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(velocity))

    def make_vocab(self) -> list[str]:
        """
        Generate a vocabulary of tokens based on pitch, duration bins, dstart bins, and velocity bins.

        Returns:
        list[str]: List of generated tokens.
        """
        vocab = []
        for it, pitch in enumerate(range(21, 109)):
            for jt in range(self.n_duration_bins):
                for kt in range(self.n_dstart_bins):
                    for vt in range(self.n_velocity_bins):
                        key = f"{kt}_{jt}_{vt}_{pitch}"
                        vocab.append(key)

        return vocab
