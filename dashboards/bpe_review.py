import datetime
from concurrent.futures import ThreadPoolExecutor

import yaml
import pandas as pd
import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import Dataset, load_dataset
from tokenizers.pre_tokenizers import PreTokenizer

from midi_trainable_tokenizers.bpe_tokenizer import BpeMidiTokenizer
from object_generators.base_tokenizer_generator import BaseTokenizerGenerator


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
    st.write(
        """
        #### First - select base_tokenizer

        This tokenizer will be used to tokenize MIDI data and turn it to text.
        A TrainableTokenzier (such as BpeTokenizer) will train on text data to compose a suitable vocabulary.
        """
    )
    tokenizer_generator = BaseTokenizerGenerator()

    base_tokenizer_names = tokenizer_generator.name_to_factory_map.keys()
    base_tokenizer_name = st.selectbox(label="tokenizer", options=base_tokenizer_names)

    st.write(tokenizer_generator.tokenizer_info(name=base_tokenizer_name))
    with st.form("base tokenizer generation"):
        base_tokenizer = tokenizer_generator.generate_tokenizer_with_streamlit(base_tokenizer_name)
        st.form_submit_button("Run")
    st.write(f"base tokenizer vocab size: {base_tokenizer.vocab_size}")

    st.write(
        """
        #### Now, select data for training

        This data will be turned into text by a base_tokenizer.
        """
    )
    dataset_names = ["roszcz/maestro-sustain-v2"]
    dataset_name = st.selectbox(label="dataset", options=dataset_names)

    buffer = tokenize_data(
        tokenizer_name=base_tokenizer_name,
        parameters=base_tokenizer.parameters,
        dataset_path=dataset_name,
    )
    text_dataset = buffer.splitlines()

    with st.expander("Example of a midi data turned into text by a base_tokenizer:"):
        st.write(text_dataset[0][:250] + " [...]")

    # initialize empty Tokenizer ...
    tokenizer = BpeMidiTokenizer(base_tokenizer=base_tokenizer)
    # and get its pre-tokenizer
    pre_tokenizer: PreTokenizer = tokenizer.tokenizer.pre_tokenizer
    pre_tokenized_data = pre_tokenizer.pre_tokenize_str(text_dataset[0])
    pre_tokenized_data = [token_info[0] for token_info in pre_tokenized_data]

    st.write(
        """
            HuggingFace tokenizers always have a pre-tokenizer. In the case of BPE model,
            it is necessary to use either WhiteSpace or ByteLevel tokenizers,
            which - by default - split text into word on spaces.
            Fortunatelly we can disable this splitting by only using ByteLevel
            and setting `use_regex=False`. ByteLevel tokenizer replaces all spaces with a chr(288)
            (G with a dot) token - this is OpenAI implementation - we could have used any seperation token.

            HuggingFace BPE algorithm is a sub-word algorithm, which means it will look for
            repetitive pattens of characters inside the words. It will not create a vocabulary
            that merges words together. That is why we want to treat several base_tokenizer tokens
            as one word in the training process.

            Unfortunatlly, treating all the text as one huge word is very inefficient and my computer cannot do it.
            Therefore, I also use Split pre-tokenizers alongside ByteLevel. They split the text, treating
            note_off event tokens and velocity+note_on tokens as seperate words
            ("VELOCITY-7 NOTE_ON_45", "NOTE_OFF_45"),
            while treating time tokens that appear between them as a singe word ("1T 1T 1T 1T").

            That way BPE algorithm merges only time tokens together.
        """
    )

    with st.expander(label="Example of a pre-tokenized text:"):
        st.write(pre_tokenized_data[:50])

    st.write(
        """
        #### BpeMidiTokenizer
        HF BpeTokenizer will compose a vocabulary of maximum size specified by vocab_size argument to HF BpeTrainer.
        Creating too large vocabulary on a small dataset will make all the different time differences seperate tokens.
        This will make multiple time tokens between note event a rarity and make it impossible for LLM to learn
        other structure than note_event_token - time_token - note_event_token - time_token - ...,
        but we can always experiment.
        """
    )
    max_vocab_size = st.number_input(label="max_vocab_size", value=500)
    tokenizer = BpeMidiTokenizer(base_tokenizer=base_tokenizer, max_vocab_size=max_vocab_size)

    # training the BPE tokenizer
    tokenizer.train_from_text_dataset(dataset=text_dataset)
    midi_dataset = load_hf_dataset(dataset_name=dataset_name, split="test")

    def save_tokenizer():
        current_datetime = datetime.datetime.now()

        # Format the datetime as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Create the filename
        path = f"dumps/tokenizers/tokenizer-{formatted_datetime}.json"

        tokenizer.save_tokenizer(path=path)

    st.write(
        """
        You can save the tokenizer and load it in tokenizer_review display mode.

        It will be saved in a file f"tokenizer-[current_datetime].json"
        """
    )
    st.button(label="save tokenizer", on_click=save_tokenizer)
    st.write(f"BpeMidiTokenizer vocab size: {tokenizer.vocab_size}")
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
