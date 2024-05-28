import yaml
import numpy as np
import pandas as pd

from midi_quantizers.quantizer import MidiQuantizer


class AbsoluteTimeQuantizer(MidiQuantizer):
    """
    Quantizer for MIDI data using bins for start time, duration, and velocity.

    Attributes:
        n_duration_bins (int): Number of bins for note durations.
        n_velocity_bins (int): Number of bins for note velocities.
        n_start_bins (int): Number of bins for note start times.
        sequence_duration (float): Duration of the sequence.
        keys (list[str]): List of quantization keys.
        duration_bin_edges (np.ndarray): Edges for duration bins.
        velocity_bin_edges (np.ndarray): Edges for velocity bins.
        start_bin_edges (np.ndarray): Edges for start bins.
        bin_to_duration (list[float]): Mapping from bins to duration values.
        bin_to_velocity (list[int]): Mapping from bins to velocity values.
        bin_to_start (list[float]): Mapping from bins to start values.
    """

    def __init__(
        self,
        n_duration_bins: int = 3,
        n_velocity_bins: int = 3,
        n_start_bins: int = 625,
        sequence_duration: float = 20.0,
    ):
        """
        Initialize the AbsoluteTimeQuantizer with specified bins for start time, duration, and velocity.

        Parameters:
        n_duration_bins (int): Number of bins for note durations. Defaults to 3.
        n_velocity_bins (int): Number of bins for note velocities. Defaults to 3.
        n_start_bins (int): Number of bins for note start times. Defaults to 625.
        sequence_duration (float): Duration of the sequence. Defaults to 20.0.
        """
        self.keys = ["pitch", "start_bin", "duration_bin", "velocity_bin"]
        self.n_velocity_bins = n_velocity_bins
        self.n_duration_bins = n_duration_bins
        self.n_start_bins = n_start_bins
        self.sequence_duration = sequence_duration
        self._build()

    def __rich_repr__(self):
        yield "AbsoluteTimeQuantizer"
        yield "n_duration_bins", self.n_duration_bins
        yield "n_velocity_bins", self.n_velocity_bins
        yield "n_start_bins", self.n_start_bins
        yield "sequence_duration", self.sequence_duration

    def _build(self):
        """
        Build the quantizer by loading bin edges and decoders for start time, duration, and velocity.
        """
        self._load_bin_edges()
        self._build_start_decoder()
        self._build_duration_decoder()
        self._build_velocity_decoder()

    def _load_bin_edges(self):
        """
        Load bin edges for start time, duration, and velocity from a YAML file.
        """
        artifacts_path = "midi_quantization_artifacts/bin_edges.yaml"
        with open(artifacts_path, "r") as f:
            bin_edges = yaml.safe_load(f)

        self.duration_bin_edges = bin_edges["duration"][self.n_duration_bins]
        self.velocity_bin_edges = bin_edges["velocity"][self.n_velocity_bins]
        self.start_bin_edges = np.linspace(start=0.0, stop=self.sequence_duration, num=self.n_start_bins)

    def _build_start_decoder(self):
        """
        Build a decoder to convert start time bins back to start time values.
        """
        self.bin_to_start = []
        for it in range(1, len(self.start_bin_edges)):
            start = (self.start_bin_edges[it - 1] + self.start_bin_edges[it]) / 2
            self.bin_to_start.append(start)

        last_start = self.start_bin_edges[-1] + self.sequence_duration / self.n_start_bins
        self.bin_to_start.append(last_start)

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

    def _build_velocity_decoder(self):
        """
        Build a decoder to convert velocity bins back to velocity values.
        """
        self.bin_to_velocity = [int(0.8 * self.velocity_bin_edges[1])]

        for it in range(2, len(self.velocity_bin_edges)):
            velocity = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(velocity))

    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantize the start time, duration, and velocity values in the DataFrame into bins.

        Parameters:
        df (pd.DataFrame): The DataFrame containing MIDI data.

        Returns:
        pd.DataFrame: The quantized DataFrame.
        """
        df["start_bin"] = np.digitize(df.start, self.start_bin_edges) - 1
        df["duration_bin"] = np.digitize(df.duration, self.duration_bin_edges) - 1
        df["velocity_bin"] = np.digitize(df.velocity, self.velocity_bin_edges) - 1
        return df

    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the quantization to the DataFrame, 
        converting bin indices to actual start time, duration, and velocity values.

        Parameters:
        df (pd.DataFrame): The DataFrame to apply quantization to.

        Returns:
        pd.DataFrame: The DataFrame with quantization applied.
        """
        df["start"] = df.start_bin.map(lambda it: self.bin_to_start[it])
        df["duration"] = df.duration_bin.map(lambda it: self.bin_to_duration[it])
        df["end"] = df.start + df.duration
        df["velocity"] = df.velocity_bin.map(lambda it: self.bin_to_velocity[it])
        return df
