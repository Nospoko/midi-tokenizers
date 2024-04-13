from abc import abstractmethod

import streamlit as st

from quantizer import MidiQuantizer
from absolute_time_quantizer import AbsoluteTimeQuantizer
from relative_time_quantizer import RelativeTimeQuantizer


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
    - `pitch`: uses all 88 pitch values\n
    - `velocity`: quantization using bins from artifacts/bin_edges.yaml\n
    - timing: \n
        - `start`: quantizes start into `n_start_bins` bins, evenly distributed across `sequence_duration`.\n
        If the piece is longer that `sequence_duration`, all late notes will land in the last bin.
        - `duration`: quantization using bins from artifacts/bin_edges.yaml\n
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
    - `pitch`: uses all 88 pitch values\n
    - `velocity`: quantization using bins from artifacts/bin_edges.yaml\n
    - timing: \n
        - `dstart`: Calculates time between consecutive notes played.
        Quantizes this time into bins specified in artifacts/bin_edges.yaml
        - `duration`: quantization using bins from artifacts/bin_edges.yaml
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


class QuantizerGenerator:
    # append new factories to this dict when new Quantizers are defined.
    name_to_factory_map: dict[str, "QuantizerFactory"] = {
        "AbsoluteTimeQuantizer": AbsoluteTimeQuantizerFactory(),
        "RelativeTimeQuantizer": RelativeTimeQuantizerFactory(),
    }

    def quantization_info(self, name: str):
        return self.name_to_factory_map[name].quantizer_desc

    def generate_quantizer_with_streamlit(self, name: str) -> MidiQuantizer:
        factory = self.name_to_factory_map[name]
        quantization_cfg = factory.select_parameters()

        return factory.create_quantizer(quantization_cfg)
