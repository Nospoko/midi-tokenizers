import json
from glob import glob

import streamlit as st

from midi_tokenizers.midi_tokenizer import MidiTokenizer
from midi_trainable_tokenizers.bpe_tokenizer import BpeMidiTokenizer
from midi_tokenizers_generation.base_tokenizer_generator import TokenizerFactory, name_to_base_factory_map


class BpeMidiTokenizerFactory(TokenizerFactory):
    tokenizer_desc = """
    This tokenizer can be trained on tokens generated by on of the no-loss models,
    which is passed to it by base_tokenizer parameter.

    WARNING: be sure to choose base-tokenizer that was used during training if using pre-trained BPE tokenizer.
    (I will think of a better serialization technique later)

    You can train new tokenizers by messing with scripts/train_bpe.py
    """

    @staticmethod
    def select_parameters() -> dict:
        trained_tokenizers_options = glob("dumps/tokenizers/*.json")
        path = st.selectbox(label="pre-trained tokenizers", options=trained_tokenizers_options)
        with open(path) as file:
            serialized_tokenizer = json.load(file)
        serialized_tokenizer["bpe_tokenizer"] = json.loads(serialized_tokenizer["bpe_tokenizer"])
        st.json(serialized_tokenizer, expanded=False)
        return {"path": path}

    @staticmethod
    def create_tokenizer(parameters: dict) -> BpeMidiTokenizer:
        return BpeMidiTokenizer.from_file(**parameters)


name_to_factory_map = name_to_base_factory_map | {"BpeMidiTokenizer": BpeMidiTokenizerFactory()}


def tokenizer_info(name: str):
    return name_to_factory_map[name].tokenizer_desc


def generate_tokenizer_with_streamlit(name: str) -> MidiTokenizer:
    factory = name_to_factory_map[name]
    parameters = factory.select_parameters()

    return factory.create_tokenizer(parameters)


def generate_tokenizer(name: str, parameters: dict) -> MidiTokenizer:
    factory = name_to_factory_map[name]
    return factory.create_tokenizer(parameters)
