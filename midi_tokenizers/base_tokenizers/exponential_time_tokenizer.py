import numpy as np
import pandas as pd

from midi_tokenizers.base_tokenizers.midi_tokenizer import MidiTokenizer


class ExponentialTimeTokenizer(MidiTokenizer):
    """
    Tokenizer for MIDI data using exponential time quantization.

    Attributes:
        min_time_unit (float): Minimum time unit for quantization (default: 0.01)
        n_velocity_bins (int): Number of velocity quantization bins (default: 128)
        n_special_ids (int): Number of reserved special token IDs (default: 1024)
        special_tokens (list[str]): Custom special tokens included in vocabulary
        first_placeholder_token (int): Index where placeholder tokens begin
        original_vocab_size (int): Size of vocabulary before adding placeholders
        token_to_id (dict): Maps token strings to their numerical IDs
        velocity_bin_edges (np.ndarray): Boundaries for velocity quantization bins
        bin_to_velocity (list[int]): Maps velocity bins back to MIDI velocity values
        name (str): Tokenizer identifier ("ExponentialTimeTokenizer")
        step_to_token (dict): Maps quantized time steps to token strings
        token_to_step (dict): Maps token strings to quantized time steps
        pad_token_id (int): ID of the padding token
    """

    def __init__(
        self,
        min_time_unit: float = 0.01,
        n_velocity_bins: int = 128,
        n_special_ids: int = 1024,
        special_tokens: list[str] = None,
    ):
        """
        Initialize the ExponentialTimeTokenizer with specified time unit, velocity bins, and special tokens.

        Parameters:
            min_time_unit: Smallest time unit for quantization, in seconds. Determines timing precision.
            n_velocity_bins: Number of bins for quantizing MIDI velocity values (1-127).
            n_special_ids: Number of reserved IDs for special tokens and placeholders.
            special_tokens: List of special token strings to include in vocabulary.
        """
        super().__init__(special_tokens)
        self.min_time_unit = min_time_unit
        self.n_velocity_bins = n_velocity_bins
        # `special_ids` should suggest "special places in the vocab"
        self.n_special_ids = n_special_ids

        # Will be changed in _build_vocab
        self.first_placeholder_token = 0
        self.original_vocab_size = 0

        self._build_vocab()
        self.velocity_bin_edges = np.linspace(0, 128, num=n_velocity_bins + 1, endpoint=True).astype(int)

        self._build_velocity_decoder()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}
        self.name = "ExponentialTimeTokenizer"

        self.step_to_token = {int(round(dt / min_time_unit)): token for dt, token in self.dt_to_token.items()}
        self.token_to_step = {token: step for step, token in self.step_to_token.items()}
        self.pad_token_id = self.token_to_id["<PAD>"]

    def __rich_repr__(self):
        yield "min_time_unit", self.min_time_unit
        yield "vocab_size", self.vocab_size
        yield "n_placeholder_tokens", self.n_special_ids

    @property
    def parameters(self):
        return {
            "min_time_unit": self.min_time_unit,
            "n_velocity_bins": self.n_velocity_bins,
            "n_special_ids": self.n_special_ids,
            "special_tokens": self.special_tokens,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def n_placeholder_tokens(self):
        return self.original_vocab_size + self.n_special_ids - self.first_placeholder_token

    def _build_vocab(self):
        """
        Build the vocabulary of the ExponentialTimeTokenizer,
        including special tokens, note tokens, velocity tokens, and time tokens.
        """
        self.vocab = ["<PAD>", "<CLS>"]

        self.token_to_velocity_bin = {}
        self.velocity_bin_to_token = {}

        self.token_to_pitch = {}
        self.pitch_to_on_token = {}
        self.pitch_to_off_token = {}

        self.token_to_dt = {}
        self.dt_to_token = []

        # Add MIDI note and velocity tokens to the vocabulary
        for pitch in range(21, 109):
            note_on_token = f"NOTE_ON_{pitch}"
            note_off_token = f"NOTE_OFF_{pitch}"

            self.vocab.append(note_on_token)
            self.vocab.append(note_off_token)

            self.token_to_pitch |= {note_on_token: pitch, note_off_token: pitch}
            self.pitch_to_on_token |= {pitch: note_on_token}
            self.pitch_to_off_token |= {pitch: note_off_token}

        for vel in range(self.n_velocity_bins):
            velocity_token = f"VELOCITY_{vel}"
            self.vocab.append(velocity_token)
            self.token_to_velocity_bin |= {velocity_token: vel}
            self.velocity_bin_to_token |= {vel: velocity_token}

        time_vocab, token_to_dt, dt_to_token = self._time_vocab()
        self.vocab += time_vocab
        self.first_placeholder_token = len(self.vocab)
        self.original_vocab_size = self.first_placeholder_token

        for it in range(self.n_special_ids):
            self.vocab.append(f"<PLACEHOLDER_{it}>")

        # add special tokens in the placeholder token place
        for special_token in self.special_tokens:
            self.vocab[self.first_placeholder_token] = special_token
            self.first_placeholder_token += 1

        self.token_to_dt = token_to_dt
        self.dt_to_token = dt_to_token
        self.max_time_value = self.token_to_dt[time_vocab[-1]]  # Maximum time

    def _time_vocab(self) -> tuple[dict, dict, dict]:
        """
        Generate time tokens and their mappings.

        Returns:
        tuple[dict, dict, dict]: The time vocabulary, token to time mapping, and time to token mapping.
        """
        time_vocab = []
        token_to_dt = {}
        dt_to_token = {}

        dt_it = 1
        dt = self.min_time_unit
        # Generate time tokens with exponential distribution
        while dt < 1:
            time_token = f"{dt_it}T"
            time_vocab.append(time_token)
            dt_to_token |= {dt: time_token}
            token_to_dt |= {time_token: dt}
            dt *= 2
            dt_it *= 2
        return time_vocab, token_to_dt, dt_to_token

    def add_special_tokens(self, special_tokens: list[str]):
        self.special_tokens += special_tokens
        for special_token in special_tokens:
            self.vocab[self.first_placeholder_token] = special_token
            self.first_placeholder_token += 1

    def quantize_frame(self, df: pd.DataFrame):
        """
        Quantize the velocity values in the DataFrame into bins.

        Parameters:
        df (pd.DataFrame): The DataFrame containing MIDI data.

        Returns:
        pd.DataFrame: The quantized DataFrame.
        """
        df = df.copy()
        df["velocity_bin"] = np.digitize(df["velocity"], self.velocity_bin_edges) - 1
        df["start"] = np.round(df["start"] / self.min_time_unit) * self.min_time_unit
        df["end"] = np.round(df["end"] / self.min_time_unit) * self.min_time_unit
        # We have to manually prevent notes with 0.0 duration after rounding
        df.loc[df["start"] == df["end"], "end"] += self.min_time_unit
        df["duration"] = df["end"] - df["start"]
        return df

    def _build_velocity_decoder(self):
        """
        Build a decoder to convert velocity bins back to velocity values.
        """
        self.bin_to_velocity = []
        for it in range(1, len(self.velocity_bin_edges)):
            velocity = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(velocity))

    @staticmethod
    def _notes_to_events(notes: pd.DataFrame) -> list[dict]:
        """
        Convert MIDI note DataFrame into a list of note-on and note-off events.

        Parameters:
        notes (pd.DataFrame): The DataFrame containing MIDI notes.

        Returns:
        list[dict]: The list of note events.
        """
        note_on_df: pd.DataFrame = notes.loc[:, ["start", "pitch", "velocity_bin"]]
        note_off_df: pd.DataFrame = notes.loc[:, ["end", "pitch", "velocity_bin"]]

        note_off_df["time"] = note_off_df["end"]
        note_off_df["event"] = "NOTE_OFF"
        note_on_df["time"] = note_on_df["start"]
        note_on_df["event"] = "NOTE_ON"

        note_on_events = note_on_df.to_dict(orient="records")
        note_off_events = note_off_df.to_dict(orient="records")
        note_events = note_off_events + note_on_events

        note_events = sorted(note_events, key=lambda event: event["time"])
        return note_events

    def _time_to_steps(self, dt: float) -> int:
        return int(round(dt / self.min_time_unit))

    def tokenize_time_distance(self, dt: float) -> list[str]:
        steps = self._time_to_steps(dt)
        time_tokens = []
        remaining_steps = steps

        # Sort step sizes in descending order
        sorted_steps = sorted(self.step_to_token.keys(), reverse=True)

        for step in sorted_steps:
            while remaining_steps >= step:
                time_tokens.append(self.step_to_token[step])
                remaining_steps -= step

        if remaining_steps > 0:
            raise ValueError(
                f"Unable to exactly tokenize time distance: {dt} ({steps} steps), remaining: {remaining_steps} steps"
            )

        return time_tokens

    def tokenize(self, notes_df: pd.DataFrame) -> list[str]:
        notes_df = self.quantize_frame(notes_df)
        notes_df = notes_df.sort_values(by="pitch", kind="stable")
        notes_df.sort_values(by="start", kind="stable")
        events = self._notes_to_events(notes_df)

        tokens = []
        previous_time = 0

        for event in events:
            dt = event["time"] - previous_time
            tokens.extend(self.tokenize_time_distance(dt))
            if event["event"] == "NOTE_ON":
                tokens.append(self.velocity_bin_to_token[event["velocity_bin"]])
                tokens.append(self.pitch_to_on_token[event["pitch"]])
            else:
                tokens.append(self.pitch_to_off_token[event["pitch"]])
            previous_time = event["time"]

        return tokens

    def untokenize(self, tokens: list[str], complete_notes: bool = False) -> pd.DataFrame:
        events = []
        current_time = 0.0
        current_velocity = 0

        for token in tokens:
            if token.endswith("T"):
                current_time += self.token_to_dt[token]
            elif token.startswith("VELOCITY"):
                current_velocity = self.bin_to_velocity[self.token_to_velocity_bin[token]]
            elif token.startswith("NOTE_ON"):
                events.append((current_time, "on", self.token_to_pitch[token], current_velocity))
            elif token.startswith("NOTE_OFF"):
                events.append((current_time, "off", self.token_to_pitch[token], 0))

        events.sort()  # Sort by time
        notes = []
        open_notes = {}

        for time, event_type, pitch, velocity in events:
            if event_type == "on":
                if pitch in open_notes:
                    # Close the previous note if it's still open
                    start, vel = open_notes.pop(pitch)
                    notes.append({"pitch": pitch, "start": start, "end": time, "velocity": vel})
                open_notes[pitch] = (time, velocity)
            elif event_type == "off":
                if pitch in open_notes:
                    start, vel = open_notes.pop(pitch)
                    notes.append({"pitch": pitch, "start": start, "end": time, "velocity": vel})

        # Close any remaining open notes
        if complete_notes:
            for pitch, (start, vel) in open_notes.items():
                notes.append({"pitch": pitch, "start": start, "end": time, "velocity": vel})

        notes_df = pd.DataFrame(notes)
        if not notes_df.empty:
            notes_df.loc[notes_df["end"] == notes_df["start"], "end"] += self.min_time_unit
            notes_df = notes_df.sort_values(by="pitch", kind="stable")
            notes_df = notes_df.sort_values("start", kind="stable").reset_index(drop=True)

        return notes_df
