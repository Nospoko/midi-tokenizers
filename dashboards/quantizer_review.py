import yaml
import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import Dataset, load_dataset

from dashboards.quantizer_generator import QuantizerGenerator


def select_dataset():
    dataset_names = ["roszcz/maestro-sustain-v2"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)

    dataset = load_dataset(dataset_name, split="test")
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
    quantizer_generator = QuantizerGenerator()

    quantizer_names = quantizer_generator.name_to_factory_map.keys()
    quantizer_name = st.selectbox(label="quantizer", options=quantizer_names)
    with st.form("quantizer generation"):
        quantizer = quantizer_generator.generate_quantizer_with_streamlit(quantizer_name)
        st.form_submit_button("Run")

    midi_dataset = select_dataset()

    record = select_record(midi_dataset=midi_dataset)

    piece = MidiPiece.from_huggingface(record=record)
    end = piece.df.start.max()

    fragment_selection_columns = st.columns(2)
    start = fragment_selection_columns[0].number_input("start", value=0.0)
    finish = fragment_selection_columns[1].number_input(f"finish [0-{end}]", value=60.0)
    piece = piece.trim(start, finish)

    quantized_piece = quantizer.quantize_piece(piece=piece)

    review_columns = st.columns(2)
    with review_columns[0]:
        st.write("Original piece")
        streamlit_pianoroll.from_fortepyan(piece=piece)
    with review_columns[1]:
        st.write("Quantized piece")
        streamlit_pianoroll.from_fortepyan(piece=quantized_piece)


if __name__ == "__main__":
    main()
