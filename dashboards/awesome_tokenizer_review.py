import datetime
from concurrent.futures import ThreadPoolExecutor

import yaml
import pandas as pd
import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import Dataset, load_dataset
from tokenizers.pre_tokenizers import PreTokenizer

from midi_trainable_tokenizers.awesome_midi_tokenzier import AwesomeMidiTokenizer
from midi_tokenizers_generation.base_tokenizer_generator import (
    tokenizer_info,
    generate_tokenizer,
    name_to_base_factory_map,
    generate_tokenizer_with_streamlit,
)


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


# logic copied from AmazingMidiTokenizer
def base_ids_to_awesome_tokens(base_token_ids: list[int]) -> list[str]:
    encryption_offset = 100
    awesome_tokens = [chr(token_id + encryption_offset) for token_id in base_token_ids]
    return awesome_tokens


# logic copied from default AmazingMidiTokenzier
@st.cache_data
def tokenize_data(tokenizer_name, parameters: dict, dataset_path: str):
    base_tokenizer = generate_tokenizer(tokenizer_name, parameters)
    train_dataset = load_dataset(dataset_path, split="test")

    def process_record(record):
        notes = pd.DataFrame(record["notes"])
        tokens = base_tokenizer.encode(notes=notes)
        awesome_tokens = base_ids_to_awesome_tokens(tokens)

        # Split tokens into chunks of less than max_token_length characters
        # Create chunks of 512 characters with a rolling window of step 256
        chunked_tokens = []
        chunk = ""
        for i in range(0, len(tokens), 32):
            chunk = "".join(str(token) for token in awesome_tokens[i : i + 64])
            chunked_tokens.append(chunk)

        # Join chunks with "\n"
        return "\n".join(chunked_tokens) + "\n"

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

    base_tokenizer_names = name_to_base_factory_map.keys()
    base_tokenizer_name = st.selectbox(label="tokenizer", options=base_tokenizer_names)

    st.write(tokenizer_info(name=base_tokenizer_name))
    with st.form("base tokenizer generation"):
        base_tokenizer = generate_tokenizer_with_streamlit(base_tokenizer_name)
        st.form_submit_button("Run")
    st.write(f"base tokenizer vocab size: {base_tokenizer.vocab_size}")

    st.write(
        """
        #### Now, select data for training

        This data will be turned into tokens by a base_tokenizer,
        then tokens will be turned into characters inside AwesomeMidiTokenizer.
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
    trainable_midi_tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer)
    # and get its pre-tokenizer
    pre_tokenizer: PreTokenizer = trainable_midi_tokenizer.text_tokenizer.pre_tokenizer
    pre_tokenized_data = pre_tokenizer.pre_tokenize_str(text_dataset[0])
    pre_tokenized_data = [token_info[0] for token_info in pre_tokenized_data]

    st.write(
        """
            HuggingFace tokenizers always have a pre-tokenizer. In the case of BPE model,
            it is necessary to use either WhiteSpace or ByteLevel tokenizers. I use WhiteSpace,
            as apart from splitting text on spaces it does not mess with my custom midi encoding.

            HuggingFace BPE algorithm is a sub-word algorithm, which means it will look for
            repetitive pattens of characters inside the words. It will not create a vocabulary
            that merges words together. That is why we want to treat several base_tokenizer tokens
            as one word in the training process, and why i made every token a single character instead.

            Unfortunatlly, treating all the text as one huge word is very inefficient and my computer cannot do it.
            Therefore, I split the data into equal-sized fragments a rolling-window style,
            with step_size = 0.5 * window size.

            It should give us a good approximation of the data.
            Window size should be equal to the maximum token length in AwesomeMidiTokenizer.
            You cannot change this parameter on dashboard as it may cause the dashboard to crash,
            you can play with it in a training script in this repo.
        """
    )

    with st.expander(label="Example of a pre-tokenized text:"):
        st.write(pre_tokenized_data[:50])

    max_vocab_size = st.number_input(label="max_vocab_size", value=10000)
    trainable_midi_tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer, max_vocab_size=max_vocab_size)

    # training the BPE tokenizer
    trainable_midi_tokenizer.train_from_text_dataset(dataset=text_dataset)
    midi_dataset = load_hf_dataset(dataset_name=dataset_name, split="test")

    def save_tokenizer():
        current_datetime = datetime.datetime.now()

        # Format the datetime as a string
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # Create the filename
        path = f"dumps/awesome_tokenizers/awesome-tokenizer-test-{formatted_datetime}.json"

        trainable_midi_tokenizer.save_tokenizer(path=path)

    st.write(
        """
        You can save the tokenizer and load it in tokenizer_review display mode.
        Note that these are test tokenziers

        It will be saved in a file "awesome_tokenizer-test-[current_datetime].json"
        """
    )
    st.button(label="save tokenizer", on_click=save_tokenizer)
    st.write(f"AwesomeMidiTokenizer vocab size: {trainable_midi_tokenizer.vocab_size}")
    st.write("### Test the tokenizer")
    record = select_record(midi_dataset=midi_dataset)
    piece = MidiPiece.from_huggingface(record=record)
    end = piece.df.start.max()

    fragment_selection_columns = st.columns(2)
    start = fragment_selection_columns[0].number_input("start", value=0.0)
    finish = fragment_selection_columns[1].number_input(f"finish [0-{end}]", value=60.0)
    piece = piece.trim(start, finish)

    tokens = trainable_midi_tokenizer.tokenize(piece.df)
    untokenized_notes = trainable_midi_tokenizer.untokenize(tokens=tokens)

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
