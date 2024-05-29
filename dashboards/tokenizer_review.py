import yaml
import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import Dataset, load_dataset

from midi_tokenizers_generation.tokenizer_generator import tokenizer_info, name_to_factory_map, generate_tokenizer_with_streamlit


@st.cache_data
def load_hf_dataset(dataset_name: str, split: str):
    return load_dataset(dataset_name, split=split)


def select_dataset():
    dataset_names = ["roszcz/maestro-sustain-v2"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)

    dataset = load_hf_dataset(dataset_name=dataset_name, split="test")
    return dataset


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


def main():
    tokenizer_names = name_to_factory_map.keys()
    tokenizer_name = st.selectbox(label="tokenizer", options=tokenizer_names)

    st.write(tokenizer_info(name=tokenizer_name))
    with st.form("tokenizer generation"):
        tokenizer = generate_tokenizer_with_streamlit(tokenizer_name)
        st.form_submit_button("Run")
    st.write(f"vocab size: {tokenizer.vocab_size}")

    midi_dataset = select_dataset()

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
