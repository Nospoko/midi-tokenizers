# MIDI Tokenizers

The `midi_tokenizers` package provides utilities to tokenize and process MIDI files for various tasks, including music generation and analysis. The package includes different tokenization and quantization methods for experiments.
### Installation

To install the package, you can clone the GitHub repository and use pip to install it:

```bash
git clone https://github.com/Nospoko/midi-tokenizers.git
cd midi-tokenizers
pip install .
```

## Package Contents

The `midi_tokenizers` package includes the following modules and classes:

- **MidiTokenizer**: Base class for all MIDI tokenizers.
- **OneTimeTokenizer**: Tokenizer that uses a single time token.
- **ExponentialTimeTokenizer**: Tokenizer that uses multiple time tokens, rising exponentially.
- **QuantizedMidiTokenizer**: Tokenizer that uses quantization to first bin the data and then treats all possible combinations as separate tokens.
- **MidiTrainableTokenizer**: Base class for trainable MIDI tokenizers.
- **BpeMidiTokenizer**: BPE-based tokenizer for MIDI data.
- **AwesomeMidiTokenizer**: MIDI tokenizer that uses BPE and encodes base tokenizer token IDs as characters.

### Dashboards

The `dashboards` module provides a set of Streamlit applications for reviewing and interacting with different tokenizers and quantizers. Each file in the `dashboards` module serves a specific purpose and can be run independently.

#### Running the Main Dashboard

```bash
PYTHONPATH=. streamlit run --server.port 4466 dashboards/main.py
```

### Dashboard Structure

- **awesome_tokenizer_review.py**: Review and interact with the Awesome MIDI Tokenizer. Allows to save traied tokenizer.
- **bpe_review.py**: Review and interact with the BPE MIDI Tokenizer. Allows to save trained tokenizer.
- **quantizer_review.py**: Review and interact with different quantization methods.
- **tokenizer_review.py**: General tokenizer review interface.
- **common/components.py**: Common components used across different dashboard files.
- **main.py**: Main entry point for the dashboard, providing a comprehensive interface to explore tokenizers and quantizers.

## Tokenization Methods

### Loading and Tokenizing MIDI Data

Let's start by loading a sample MIDI dataset and tokenize it using different tokenizers.

```python
import pandas as pd
from midi_tokenizers import OneTimeTokenizer, ExponentialTimeTokenizer, QuantizedMidiTokenizer

# Sample MIDI data
data = pd.DataFrame({
    'pitch': [74, 71, 83, 79, 77],
    'velocity': [92, 110, 103, 92, 89],
    'start': [0.973958, 0.985417, 0.985417, 0.989583, 0.989583],
    'end': [2.897917, 2.897917, 2.897917, 2.897917, 2.897917],
    'duration': [1.923958, 1.912500, 1.912500, 1.908333, 1.908333]
})
```

### Exponential Time Tokenizer

The `ExponentialTimeTokenizer` is a specialized tokenizer that converts MIDI data into a sequence of tokens using an exponential time encoding scheme. This tokenizer is designed to handle musical sequences by breaking down each note into discrete events and encoding the time differences between them.
For example, for `min_time_unit=0.01`, time token values are:
```py
{
    "1T": "10ms",
    "2T": "20ms",
    "3T": "40ms",
    "4T": "80ms",
    "5T": "160ms",
    "6T": "320ms",
    "7T": "640ms",
}
```

#### How It Works

1. **Data Representation**:

    | pitch | velocity |   start   |     end    |
    |-------|----------|-----------|------------|
    |   59  |    94    |  0.000000 |  0.072727  |
    |   48  |    77    |  0.077273 |  0.177273  |
    |   60  |    95    |  0.102273 |  0.229545  |
    |   47  |    79    |  0.159091 |  0.275000  |
   - Each row in the original DataFrame represents one note played.
   - The tokenization process involves dividing each note into two separate events: `note_on` and `note_off`.
   - Each event is described by its type (either `note_on` or `note_off`), key number, velocity, and time (original start time for `note_on` events and original end time for `note_off` events).

    | pitch | velocity |      time | event_type |
    |-------|----------|-----------|------------|
    |    59 |       94 |  0.000000 |    note_on |
    |    59 |       94 |  0.072727 |   note_off |
    |    48 |       77 |  0.077273 |    note_on |
    |    60 |       95 |  0.102273 |    note_on |
    |    47 |       79 |  0.159091 |    note_on |
    |    48 |       77 |  0.177273 |   note_off |
    |    60 |       95 |  0.229545 |   note_off |
    |    47 |       79 |  0.275000 |   note_off |

2. **Velocity Encoding**:
   - The process starts by transcribing the velocity value of the note.
   - The velocity is quantized into a finite number of bins and encoded as a token.

   ['VELOCITY_94']

3. **Event Type Encoding**:
   - Depending on the event type (`note_on` or `note_off`), an event token is added to indicate which key was pressed or released.

   ['VELOCITY_94', 'NOTE_ON_59']
4. **Time Difference Encoding**:
   - The time difference between two consecutive events is encoded using an exponential scheme.

    ['VELOCITY_94', 'NOTE_ON_59', '4T']
5. **Event Sequencing**:
   - The process moves on to the next event, calculating the subsequent time difference and encoding it.
   - If the next time difference is less than a defined `min_time_unit` (e.g., 10ms), no time token is added.

   ['VELOCITY_94', 'NOTE_ON_59', '4T', 'VELOCITY_94', 'NOTE_OFF_59']

6. **Token Sequence Construction**:
   - This process continues until the entire musical sequence is represented in a text format with a discrete number of tokens.
   - The resulting token sequence preserves the musical information with minimal loss, ensuring that the encoded and decoded sequences sound almost identical.

#### Example

Let's illustrate the tokenization process with a simple example. Given a DataFrame with the following MIDI data:

```python
import pandas as pd

# Sample MIDI data
data = pd.DataFrame({
    'pitch': [59, 48, 60, 47],
    'velocity': [94, 77, 95, 79],
    'start': [0.000000, 0.077273, 0.102273, 0.159091],
    'end': [0.072727, 0.177273, 0.229545, 0.275000]
})

# Initialize the Exponential Time Tokenizer
from midi_tokenizers import ExponentialTimeTokenizer
exp_time_tokenizer = ExponentialTimeTokenizer()

# Tokenize the sample data
tokens = exp_time_tokenizer.tokenize(data)

print(tokens)
```

### Output
The output tokens might look like this:

```
['VELOCITY_94', 'NOTE_ON_59', '4T', 'VELOCITY_94', 'NOTE_OFF_59', 'VELOCITY_77', 'NOTE_ON_48', '2T', 'VELOCITY_95', 'NOTE_ON_60', '3T', '2T', 'VELOCITY_79', 'NOTE_ON_47', '2T', 'VELOCITY_77', 'NOTE_OFF_48', 'VELOCITY_97', 'NOTE_ON_59', '3T']
```

In this example, the tokens represent the time intervals (`1T`, `2T`), velocities (`VELOCITY_92`, `VELOCITY_110`, etc.), and the note events (`NOTE_ON_74`, `NOTE_OFF_74`, etc.).

### Benefits

The `ExponentialTimeTokenizer` provides an efficient and compact representation of MIDI data, preserving the essential musical information while minimizing data redundancy.
### One-Time Tokenizer
This tokenizer works exactly like ExponentialTimeTokenizer but has only one time token.

```python
# Initialize the One-Time Tokenizer
one_time_tokenizer = OneTimeTokenizer()

# Tokenize the sample data
tokens = one_time_tokenizer.tokenize(sample_data)

print(tokens)
```

### Quantized MIDI Tokenizer

```python
# Initialize the Quantized MIDI Tokenizer
quantized_tokenizer = QuantizedMidiTokenizer()

# Tokenize the sample data
tokens = quantized_tokenizer.tokenize(sample_data)

print(tokens)
```

### Awesome MIDI Tokenizer

The `AwesomeMidiTokenizer` is a MIDI tokenizer that uses Byte-Pair Encoding (BPE) and encodes base tokenizer token IDs as characters. This tokenizer is designed for efficient tokenization and high-quality representation of MIDI data.

#### BPE on MIDI Data

When applying BPE to MIDI data, the process involves several steps to convert the MIDI notes into a format suitable for BPE tokenization.

1. **Dump Notes into Text**:
   - First, the notes are dumped into a glob of text.
   - This text is then used to train a typical text BPE tokenizer using the Hugging Face `tokenizers` library.

2. **Byte-Pair Encoding (BPE)**:
   - BPE is used in NLP to minimize the vocabulary that the model uses. For MIDI vocabulary, which contains about 219 tokens, BPE can expand the vocabulary and minimize the number of tokens needed to describe the data.
   - The vocabulary of an Awesome tokenizer consists of words created by merging several `ExponentialTimeTokens`, representing the most common sequences in the training data.
   - This provides models with more context without significantly increasing the input size, making training easier.

3. **Tokenization Process**:
   - **Generate Tokens**: The MIDI files are first tokenized using an `ExponentialTimeTokenizer`.
   - **Convert to Unicode Characters**: Each distinct token is transformed into a unicode character.
   - **Pre-tokenization**: Just like in NLP, the text is split into "words" to manage computational complexity. This step is crucial as it segments the text into manageable chunks for BPE.

4. **Splitting into "Words"**:
   - BPE is a subword segmentation algorithm that encodes rare and unknown words as sequences of subword units.
   - The subword units in this context are merges of original `ExponentialTimeTokenizer` tokens, represented as unicode characters.
   - The segmentation can be done by identifying quiet moments in the music or by creating equal-sized overlapping words.

5. **Train Hugging Face BPE Tokenizer**:
   - The tokenizer is trained on the segmented text to learn common sequences and create a compact representation of the MIDI data.

Here is an example demonstrating the process:

```python
from midi_trainable_tokenizers import AwesomeMidiTokenizer
from midi_tokenizers import ExponentialTimeTokenizer
from datasets import load_dataset
import pandas as pd

# Sample MIDI data
data = pd.DataFrame({
    'pitch': [59, 48, 60, 47],
    'velocity': [94, 77, 95, 79],
    'start': [0.000000, 0.077273, 0.102273, 0.159091],
    'end': [0.072727, 0.177273, 0.229545, 0.275000]
})

# Initialize the base tokenizer
base_tokenizer = ExponentialTimeTokenizer()

# Initialize the Awesome MIDI Tokenizer
tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer)

# Load MIDI dataset
dataset = load_dataset("roszcz/maestro-v2", split="train")

# Train the tokenizer
tokenizer.train(dataset)

# Tokenize the sample data
tokens = tokenizer.tokenize(data)

print(tokens)
```

This example demonstrates how to use the `AwesomeMidiTokenizer` to tokenize a sample MIDI data. The tokenizer first needs to be trained on a dataset before it can be used to tokenize new data. The training process uses the `ExponentialTimeTokenizer` as a base tokenizer and trains the BPE tokenizer on the specified dataset. After training, the tokenizer can convert new MIDI data into a sequence of tokens.

This process ensures efficient encoding of MIDI data with minimal loss of information, making it suitable for applications in music generation.


### BPE MIDI Tokenizer
Like Awesome Tokenizer, but without converting to unicode and only merges time tokens.
```python
from midi_trainable_tokenizers import BpeMidiTokenizer

# Initialize the base tokenizer
base_tokenizer = ExponentialTimeTokenizer()

# Initialize the Awesome MIDI Tokenizer
tokenizer = BpeMidiTokenizer(base_tokenizer=base_tokenizer)

# Load MIDI dataset
dataset = load_dataset("roszcz/maestro-sustain-v2", split="train")

# Train tokenzier
tokenizer.train(dataset)

tokens = tokenzier.tokenize(data)

print(tokens)
```

## Tokenization and Quantization Methods

### Tokenization

Tokenization is the process of converting MIDI data into a sequence of tokens that can be processed by machine learning models. Different tokenizers use different strategies to represent the MIDI data as tokens.

### Quantization

Quantization is the process of binning continuous MIDI data (such as note start times and velocities) into discrete values. This allows representing MIDI data in the form of tokens.

###

## Saving and Loading Tokenizers

Tokenizers can be saved to disk and loaded back when needed. This allows you to train a tokenizer once and reuse it without retraining.

### Saving a Tokenizer

```python
# Save the tokenizer
bpe_tokenizer.save_tokenizer('bpe_tokenizer.json')
```

### Loading a Tokenizer

```python
# Load the tokenizer
loaded_tokenizer = BpeMidiTokenizer.from_file('bpe_tokenizer.json')
```

## Contributing

Contributions are welcome! If you have any issues or feature requests, please open an issue on the [GitHub repository](https://github.com/Nospoko/midi-quantizers).

## Code Style

This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```
