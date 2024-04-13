from abc import abstractmethod

import streamlit as st

from quantizer import MidiQuantizer
from absolute_time_quantizer import AbsoluteTimeQuantizer
from relative_time_quantizer import RelativeTimeQuantizer


class QuantizerFactory:
    @abstractmethod
    def select_parameters() -> dict:
        pass

    @abstractmethod
    def create_quantizer(quantization_cfg: dict) -> MidiQuantizer:
        pass


class AbsoluteTimeQuantizerFactory(QuantizerFactory):
    def select_parameters():
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)
        n_duration_bins = st.number_input(label="n_duration_bins", value=3)
        n_start_bins = st.number_input(label="n_start_bins", value=625)
        sequence_duration = st.number_input(label="sequence_duration", value=20)
        quantization_cfg = {
            "n_velocity_bins": n_velocity_bins,
            "n_duration_bins": n_duration_bins,
            "n_start_bins": n_start_bins,
            "sequence_duration": sequence_duration,
        }

        return quantization_cfg

    def create_quantizer(quantization_cfg: dict) -> AbsoluteTimeQuantizer:
        return AbsoluteTimeQuantizer(**quantization_cfg)


class RelativeTimeQuantizerFactory(QuantizerFactory):
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

    def create_quantizer(quantization_cfg: dict) -> RelativeTimeQuantizer:
        return RelativeTimeQuantizer(**quantization_cfg)


class QuantizerGenerator:
    name_to_factory_map: dict[str, "QuantizerFactory"] = {
        "AbsoluteTimeQuantizer": AbsoluteTimeQuantizerFactory(),
        "RelativeTimeQuantizer": RelativeTimeQuantizerFactory(),
    }

    def generate_quantizer_streamlit(self, name: str):
        factory = self.name_to_factory_map[name]
        quantization_cfg = factory.select_parameters()

        return factory.create_quantizer(quantization_cfg)
