from abc import abstractmethod

import streamlit as st

from quantizer_generator import QuantizerGenerator
from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_tokenizers.quantized_midi_tokenizer import QuantizedMidiTokenizer


class TokenizerFactory:
    """
    Base class for Quantizer objects factory. Makes adding new Quantizers to the dashboard easier.
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
    def select_parameters():
        quantizer_generator = QuantizerGenerator()
        quantizer_name = st.selectbox(label="quantizer", options=quantizer_generator.name_to_factory_map.keys())
        factory = quantizer_generator.name_to_factory_map[quantizer_name]
        quantization_cfg = factory.select_parameters()

        return {"quantization_cfg": quantization_cfg, "quantizer_name": quantizer_name}

    @staticmethod
    def create_tokenizer(parameters: dict) -> QuantizedMidiTokenizer:
        return QuantizedMidiTokenizer(**parameters)


class TokenizerGenerator:
    # append new factories to this dict when new Tokenizers are defined.
    name_to_factory_map: dict[str, "TokenizerFactory"] = {
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
