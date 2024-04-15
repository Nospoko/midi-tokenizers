from concurrent.futures import ThreadPoolExecutor

import yaml
import pandas as pd
import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import Dataset, load_dataset
from tokenizers.pre_tokenizers import PreTokenizer

from base_tokenizer_generator import BaseTokenizerGenerator
from midi_trainable_tokenizers.bpe_tokenizer import BpeTokenizer


@st.cache_data
def load_hf_dataset(dataset_name: str, split: str):
    return load_dataset(dataset_name, split=split)


def select_record(midi_dataset: Dataset):
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()

    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=3,
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
    )

    st.write(selected_title)

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset[0]


@st.cache_data
def tokenize_data(tokenizer_name, parameters: dict, dataset_path: str):
    base_tokenizer_generator = BaseTokenizerGenerator()
    base_tokenizer = base_tokenizer_generator.generate_tokenizer(tokenizer_name, parameters)

    train_dataset = load_dataset(dataset_path, split="train")

    def process_record(record):
        notes = pd.DataFrame(record["notes"])
        tokens = base_tokenizer.tokenize(notes=notes)
        return " ".join(str(token) for token in tokens) + "\n"

    buffer = ""
    with ThreadPoolExecutor() as executor:
        # Process records concurrently
        for result in executor.map(process_record, train_dataset):
            buffer += result
    return buffer


def main():
    st.write("#### First - select base_tokenizer")
    tokenizer_generator = BaseTokenizerGenerator()

    base_tokenizer_names = tokenizer_generator.name_to_factory_map.keys()
    base_tokenizer_name = st.selectbox(label="tokenizer", options=base_tokenizer_names)

    st.write(tokenizer_generator.tokenizer_info(name=base_tokenizer_name))
    with st.form("base tokenizer generation"):
        base_tokenizer = tokenizer_generator.generate_tokenizer_with_streamlit(base_tokenizer_name)
        st.form_submit_button("Run")
    st.write(f"base tokenizer vocab size: {base_tokenizer.vocab_size}")

    st.write("#### Now, select data for training")
    dataset_names = ["roszcz/maestro-sustain-v2"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)

    buffer = tokenize_data(
        tokenizer_name=base_tokenizer_name,
        parameters=base_tokenizer.parameters,
        dataset_path=dataset_name,
    )
    text_dataset = buffer.splitlines()

    with st.expander("Example of a record turned into text by a base_tokenizer:"):
        st.write(text_dataset[0])

    # initialize empty Tokenizer
    tokenizer = BpeTokenizer(base_tokenizer=base_tokenizer)

    pre_tokenizer: PreTokenizer = tokenizer.tokenizer.pre_tokenizer
    pre_tokenized_data = pre_tokenizer.pre_tokenize_str(text_dataset[0])

    st.write(
        """
            This is done during training step on text data.
            The weird G tokens replace spaces. (It happenes inside ByteLever pre-tokenizer)
        """
    )

    with st.expander(label="Example of a record pre-tokenized by pre-tokenizers inside the huggingface tokenizer:"):
        st.write(pre_tokenized_data)

    # training the BPE tokenizer
    tokenizer.train_from_text_dataset(dataset=text_dataset)
    midi_dataset = load_hf_dataset(dataset_name=dataset_name, split="test")

    st.write("### Test the tokenizer")
    record = select_record(midi_dataset=midi_dataset)
    piece = MidiPiece.from_huggingface(record=record)
    end = piece.df.start.max()

    fragment_selection_columns = st.columns(2)
    start = fragment_selection_columns[0].number_input("start", value=0.0)
    finish = fragment_selection_columns[1].number_input(f"finish [0-{end}]", value=60.0)
    piece = piece.trim(start, finish)

    tokens = tokenizer.tokenize(piece.df)
    untokenized_notes = tokenizer.untokenize(tokens=tokens)

    untokenized_piece = MidiPiece(df=untokenized_notes, source=piece.source | {"tokenized": True})

    review_columns = st.columns(2)
    with review_columns[0]:
        st.write("Original piece")
        streamlit_pianoroll.from_fortepyan(piece=piece)
    with review_columns[1]:
        st.write("Untokenized piece")
        streamlit_pianoroll.from_fortepyan(piece=untokenized_piece)

    st.write(f"number of tokens: {len(tokens)}")

    with st.expander(label="tokens"):
        st.write(tokens)


if __name__ == "__main__":
    main()
