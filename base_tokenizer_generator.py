from abc import abstractmethod

import streamlit as st

from quantizer_generator import QuantizerGenerator
from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_tokenizers.no_loss_tokenizer import NoLossTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from midi_tokenizers.quantized_midi_tokenizer import QuantizedMidiTokenizer


class TokenizerFactory:
    """
    Base class for Tokenizer objects factory. Makes adding new Tokenizers to the dashboard easier.
    """

    tokenizer_desc = ""

    @abstractmethod
    def select_parameters() -> dict:
        pass

    @abstractmethod
    def create_tokenizer(parameters: dict) -> MidiTokenizer:
        pass


class QuantizedMidiTokenizerFactory(TokenizerFactory):
    tokenizer_desc = """
    Tokenizer that uses MidiQuantizers to first quantize the data into bins,
    then treat all possible combinations as seperate tokens.

    The tokens are stuctured like "pitch-[start_bin/dstart_bin]-duration_bin-velocity_bin"
    """

    @staticmethod
    def select_parameters() -> dict:
        quantizer_generator = QuantizerGenerator()
        quantizer_name = st.selectbox(label="quantizer", options=quantizer_generator.name_to_factory_map.keys())
        factory = quantizer_generator.name_to_factory_map[quantizer_name]
        quantization_cfg = factory.select_parameters()

        return {"quantization_cfg": quantization_cfg, "quantizer_name": quantizer_name}

    @staticmethod
    def create_tokenizer(parameters: dict) -> QuantizedMidiTokenizer:
        return QuantizedMidiTokenizer(**parameters)


class NoLossTokenizerFactory(TokenizerFactory):
    tokenizer_desc = """
    This tokenizer uses multiple time tokens, rising exponentialy from `eps` to 2 seconds.

    Quantizes velocity into `n_velocity_bins` linearly spread bins.
    """

    @staticmethod
    def select_parameters() -> dict:
        eps = st.number_input(label="eps - minimal time shift value", value=0.01, format="%0.3f")
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=32)
        return {"eps": eps, "n_velocity_bins": n_velocity_bins}

    @staticmethod
    def create_tokenizer(parameters: dict) -> NoLossTokenizer:
        return NoLossTokenizer(**parameters)


class OneTimeTokenizerFactory(TokenizerFactory):
    tokenizer_desc = """
    This tokenizer uses a single time token and uses it as many times as it needs.

    Quantizes velocity into `n_velocity_bins` linearly spread bins.
    """

    @staticmethod
    def select_parameters() -> dict:
        eps = st.number_input(label="eps - minimal time shift value", value=0.01, format="%0.3f")
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=32)
        return {"eps": eps, "n_velocity_bins": n_velocity_bins}

    @staticmethod
    def create_tokenizer(parameters: dict) -> NoLossTokenizer:
        return OneTimeTokenizer(**parameters)


class BaseTokenizerGenerator:
    # append new factories to this dict when new Tokenizers are defined.
    name_to_factory_map: dict[str, "TokenizerFactory"] = {
        "NoLossTokenizer": NoLossTokenizerFactory(),
        "OneTimeTokenizer": OneTimeTokenizerFactory(),
        "QuantizedMidiTokenizer": QuantizedMidiTokenizerFactory(),
    }

    def tokenizer_info(self, name: str):
        return self.name_to_factory_map[name].tokenizer_desc

    def generate_tokenizer_with_streamlit(self, name: str) -> MidiTokenizer:
        factory = self.name_to_factory_map[name]
        parameters = factory.select_parameters()

        return factory.create_tokenizer(parameters)

    def generate_tokenizer(self, name: str, parameters: dict) -> MidiTokenizer:
        factory = self.name_to_factory_map[name]
        return factory.create_tokenizer(parameters)
