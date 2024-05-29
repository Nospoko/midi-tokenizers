from abc import abstractmethod

import streamlit as st

from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_tokenizers.one_time_tokenizer import OneTimeTokenizer
from midi_tokenizers.no_loss_tokenizer import ExponentialTimeTokenizer
from midi_tokenizers.quantized_midi_tokenizer import QuantizedMidiTokenizer
from midi_tokenizers_generation.quantizer_generator import name_to_quantizer_factory_map


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
        quantizer_name = st.selectbox(label="quantizer", options=name_to_quantizer_factory_map.keys())
        factory = name_to_quantizer_factory_map[quantizer_name]
        quantization_cfg = factory.select_parameters()

        return {"quantization_cfg": quantization_cfg, "quantizer_name": quantizer_name}

    @staticmethod
    def create_tokenizer(parameters: dict) -> QuantizedMidiTokenizer:
        return QuantizedMidiTokenizer(**parameters)


class ExponentialTimeTokenizerFactory(TokenizerFactory):
    tokenizer_desc = """
    This tokenizer uses multiple time tokens, rising exponentialy from `eps` to 1 seconds.

    Quantizes velocity into `n_velocity_bins` linearly spread bins.
    """

    @staticmethod
    def select_parameters() -> dict:
        min_time_unit = st.number_input(label="eps - minimal time shift value", value=0.01, format="%0.3f")
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=32)
        return {"min_time_unit": min_time_unit, "n_velocity_bins": n_velocity_bins}

    @staticmethod
    def create_tokenizer(parameters: dict) -> ExponentialTimeTokenizer:
        return ExponentialTimeTokenizer(**parameters)


class OneTimeTokenizerFactory(TokenizerFactory):
    tokenizer_desc = """
    This tokenizer uses a single time token and appends it as many times as it needs.

    Quantizes velocity into `n_velocity_bins` linearly spread bins.
    """

    @staticmethod
    def select_parameters() -> dict:
        min_time_unit = st.number_input(label="eps - time shift value", value=0.01, format="%0.3f")
        n_velocity_bins = st.number_input(label="n_velocity_bins", value=32)
        return {"min_time_unit": min_time_unit, "n_velocity_bins": n_velocity_bins}

    @staticmethod
    def create_tokenizer(parameters: dict) -> ExponentialTimeTokenizer:
        return OneTimeTokenizer(**parameters)


# append new factories to this dict when new Tokenizers are defined.
name_to_base_factory_map: dict[str, "TokenizerFactory"] = {
    "ExponentialTimeTokenizer": ExponentialTimeTokenizerFactory(),
    "OneTimeTokenizer": OneTimeTokenizerFactory(),
    "QuantizedMidiTokenizer": QuantizedMidiTokenizerFactory(),
}


def tokenizer_info(name: str):
    """
    Get the description of the tokenizer.

    Parameters:
        name (str): Name of the tokenizer.

    Returns:
        str: Description of the tokenizer.
    """
    return name_to_base_factory_map[name].tokenizer_desc


def generate_tokenizer_with_streamlit(name: str) -> MidiTokenizer:
    """
    Generate a tokenizer with parameters selected using Streamlit widgets.

    Parameters:
        name (str): Name of the tokenizer.

    Returns:
        MidiTokenizer: Instance of the created tokenizer.
    """
    factory = name_to_base_factory_map[name]
    parameters = factory.select_parameters()
    return factory.create_tokenizer(parameters)


def generate_tokenizer(name: str, parameters: dict) -> MidiTokenizer:
    """
    Generate a tokenizer with the given parameters.

    Parameters:
        name (str): Name of the tokenizer.
        parameters (dict): Dictionary of parameters for the tokenizer.

    Returns:
        MidiTokenizer: Instance of the created tokenizer.
    """
    factory = name_to_base_factory_map[name]
    return factory.create_tokenizer(parameters)
