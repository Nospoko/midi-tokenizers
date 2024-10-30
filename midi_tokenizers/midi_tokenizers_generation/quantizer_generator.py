from abc import abstractmethod

import streamlit as st

from midi_tokenizers.midi_quantizers.quantizer import MidiQuantizer
from midi_tokenizers.midi_quantizers.absolute_time_quantizer import AbsoluteTimeQuantizer
from midi_tokenizers.midi_quantizers.relative_time_quantizer import RelativeTimeQuantizer


class QuantizerFactory:
    """
    Base class for Quantizer objects factory. Makes adding new Quantizers to the dashboard easier.
    """

    quantizer_desc = ""

    @abstractmethod
    def select_parameters() -> dict:
        pass

    @abstractmethod
    def create_quantizer(quantization_cfg: dict) -> MidiQuantizer:
        pass


class AbsoluteTimeQuantizerFactory(QuantizerFactory):
    quantizer_desc = """
    midi_quantization_artifacts/bin_edges.yaml stores bin edges calculated using quantiles -
    notes in maestro dataset are evenly distributed into each bin.

    - `pitch`: uses all 88 pitch values
    - `velocity`: quantization using bins from midi_quantization_artifacts/bin_edges.yaml
    - timing:
        - `start`: quantizes start into `n_start_bins` bins, evenly distributed across `sequence_duration`.
        If the piece is longer that `sequence_duration`, all late notes will land in the last bin.
        - `duration`: quantization using bins from midi_quantization_artifacts/bin_edges.yaml
    """

    @staticmethod
    def select_parameters():
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)
        n_duration_bins = st.number_input(label="n_duration_bins", value=3)
        n_start_bins = st.number_input(label="n_start_bins", value=120)
        sequence_duration = st.number_input(
            label="sequence_duration - preferably the same as fragment duration (this is a bit stupid)", value=60
        )
        quantization_cfg = {
            "n_velocity_bins": n_velocity_bins,
            "n_duration_bins": n_duration_bins,
            "n_start_bins": n_start_bins,
            "sequence_duration": sequence_duration,
        }

        return quantization_cfg

    @staticmethod
    def create_quantizer(quantization_cfg: dict) -> AbsoluteTimeQuantizer:
        return AbsoluteTimeQuantizer(**quantization_cfg)


class RelativeTimeQuantizerFactory(QuantizerFactory):
    quantizer_desc = """
    midi_quantization_artifacts/bin_edges.yaml stores bin edges calculated using quantiles -
    notes in maestro dataset are evenly distributed into each bin.

    - `pitch`: uses all 88 pitch values
    - `velocity`: quantization using bins from midi_quantization_artifacts/bin_edges.yaml
    - timing:
        - `dstart`: Calculates time between consecutive notes played.
        Quantizes this time into bins specified in midi_quantization_artifacts/bin_edges.yaml
        - `duration`: quantization using bins from midi_quantization_artifacts/bin_edges.yaml
    """

    @staticmethod
    def select_parameters() -> dict:
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)
        n_duration_bins = st.number_input(label="n_duration_bins", value=3)
        n_dstart_bins = st.number_input(label="n_dstart_bins", value=3)
        quantization_cfg = {
            "n_velocity_bins": n_velocity_bins,
            "n_duration_bins": n_duration_bins,
            "n_dstart_bins": n_dstart_bins,
        }

        return quantization_cfg

    @staticmethod
    def create_quantizer(quantization_cfg: dict) -> RelativeTimeQuantizer:
        return RelativeTimeQuantizer(**quantization_cfg)


# append new factories to this dict when new Quantizers are defined.
name_to_quantizer_factory_map: dict[str, "QuantizerFactory"] = {
    "AbsoluteTimeQuantizer": AbsoluteTimeQuantizerFactory(),
    "RelativeTimeQuantizer": RelativeTimeQuantizerFactory(),
}


def quantization_info(name: str):
    """
    Get the description of the quantizer.

    Parameters:
        name (str): Name of the quantizer.

    Returns:
        str: Description of the quantizer.
    """
    return name_to_quantizer_factory_map[name].quantizer_desc


def generate_quantizer_with_streamlit(name: str) -> MidiQuantizer:
    """
    Generate a quantizer with parameters selected using Streamlit widgets.

    Parameters:
        name (str): Name of the quantizer.

    Returns:
        MidiQuantizer: Instance of the created quantizer.
    """
    factory = name_to_quantizer_factory_map[name]
    quantization_cfg = factory.select_parameters()
    return factory.create_quantizer(quantization_cfg)


def generate_quantizer(name: str, quantization_cfg: dict) -> MidiQuantizer:
    """
    Generate a quantizer with the given parameters.

    Parameters:
        name (str): Name of the quantizer.
        quantization_cfg (dict): Dictionary of configuration parameters for the quantizer.

    Returns:
        MidiQuantizer: Instance of the created quantizer.
    """
    factory = name_to_quantizer_factory_map[name]
    return factory.create_quantizer(quantization_cfg)
