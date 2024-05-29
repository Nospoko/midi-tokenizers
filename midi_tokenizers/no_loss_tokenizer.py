import re

import numpy as np
import pandas as pd

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class ExponentialTimeTokenizer(MidiTokenizer):
    """
    Tokenizer for MIDI data using exponential time quantization.

    Attributes:
        min_time_unit (float): The minimum time unit for quantizing time.
        n_velocity_bins (int): The number of velocity bins.
        special_tokens (list[str]): A list of special tokens.
        token_to_id (dict): Mapping from tokens to their IDs.
        velocity_bin_edges (np.ndarray): Edges for velocity bins.
        bin_to_velocity (list[int]): Mapping from bins to velocity values.
        name (str): Name of the tokenizer.
    """

    def __init__(
        self,
        min_time_unit: float = 0.01,
        n_velocity_bins: int = 128,
        special_tokens: list[str] = None,
    ):
        """
        Initialize the ExponentialTimeTokenizer with specified time unit, velocity bins, and special tokens.

        Parameters:
        min_time_unit (float): The minimum time unit for quantizing time. Defaults to 0.001.
        n_velocity_bins (int): The number of velocity bins. Defaults to 128.
        special_tokens (list[str]): A list of special tokens. Defaults to None.
        """
        super().__init__(special_tokens=special_tokens)
        self.min_time_unit = min_time_unit
        self.n_velocity_bins = n_velocity_bins
        self._build_vocab()

        self.velocity_bin_edges = np.linspace(0, 127, num=n_velocity_bins + 1, endpoint=True).astype(int)

        self._build_velocity_decoder()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}
        self.name = "ExponentialTimeTokenizer"

    def __rich_repr__(self):
        yield "ExponentialTimeTokenizer"
        yield "min_time_unit", self.min_time_unit
        yield "vocab_size", self.vocab_size

    @property
    def parameters(self):
        return {"min_time_unit": self.min_time_unit, "n_velocity_bins": self.n_velocity_bins}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _build_vocab(self):
        """
        Build the vocabulary of the ExponentialTimeTokenizer,
        including special tokens, note tokens, velocity tokens, and time tokens.
        """
        self.vocab = list(self.special_tokens)

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
            dt_it += 1
        return time_vocab, token_to_dt, dt_to_token

    def quantize_frame(self, df: pd.DataFrame):
        """
        Quantize the velocity values in the DataFrame into bins.

        Parameters:
        df (pd.DataFrame): The DataFrame containing MIDI data.

        Returns:
        pd.DataFrame: The quantized DataFrame.
        """
        df["velocity_bin"] = np.digitize(df["velocity"], self.velocity_bin_edges) - 1
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

    def tokenize_time_distance(self, dt: float) -> list[str]:
        # Try filling the time beginning with the largest step
        current_step = self.max_time_value

        time_tokens = []
        filling_dt = 0
        current_step = self.max_time_value
        while True:
            if abs(dt - filling_dt) < self.min_time_unit:
                # Exit the loop when the gap is filled
                break
            if filling_dt + current_step - dt > self.min_time_unit:
                # Select time step that will fit into the gap
                current_step /= 2
            else:
                # Fill the gap with current time token
                time_token = self.dt_to_token[current_step]
                time_tokens.append(time_token)
                filling_dt += current_step

        return time_tokens

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        """
        Convert a time difference into a sequence of time tokens.

        Parameters:
        dt (float): The time difference to convert.

        Returns:
        list[str]: The list of time tokens.
        """
        notes = self.quantize_frame(notes)
        tokens = []
        # Time difference between current and previous events
        previous_time = 0
        note_events = self._notes_to_events(notes=notes)

        for current_event in note_events:
            # Calculate the time difference between current and previous event
            dt = current_event["time"] - previous_time

            # Fill the time gap
            time_tokens = self.tokenize_time_distance(dt=dt)
            tokens += time_tokens

            event_type = current_event["event"]

            # Append note event tokens
            velocity = int(current_event["velocity_bin"])
            tokens.append(self.velocity_bin_to_token[velocity])
            pitch = int(current_event["pitch"])
            if event_type == "NOTE_ON":
                tokens.append(self.pitch_to_on_token[pitch])
            else:
                tokens.append(self.pitch_to_off_token[pitch])

            previous_time = current_event["time"]

        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        note_on_events = []
        note_off_events = []

        current_time = 0
        current_velocity = 0
        for token in tokens:
            if re.search(".T$", token) is not None:
                dt: float = self.token_to_dt[token]
                current_time += dt
            if "VELOCITY" in token:
                # velocity should always be right before NOTE_ON token
                current_velocity_bin = self.token_to_velocity_bin[token]
                current_velocity: int = self.bin_to_velocity[current_velocity_bin]
            if "NOTE_ON" in token:
                note = {
                    "pitch": self.token_to_pitch[token],
                    "start": current_time,
                    "velocity": current_velocity,
                }
                note_on_events.append(note)
            if "NOTE_OFF" in token:
                note = {
                    "pitch": self.token_to_pitch[token],
                    "end": current_time,
                }
                note_off_events.append(note)

        # Both should be sorted by time right now
        note_on_events = pd.DataFrame(note_on_events)
        note_off_events = pd.DataFrame(note_off_events)

        # So if we group them by pitch ...
        pitches = note_on_events["pitch"].unique()
        note_groups = []

        for pitch in pitches:
            note_offs = note_off_events[note_off_events["pitch"] == pitch].copy().reset_index(drop=True)
            note_ons = note_on_events[note_on_events["pitch"] == pitch].copy().reset_index(drop=True)

            # and ignore (drop) all unmatched NOTE_OFF events ...
            for index, row in note_offs.iterrows():
                if row["end"] > note_ons.iloc[0]["start"]:
                    break
                else:
                    note_offs.drop(index, inplace=True)
            note_offs = note_offs.reset_index(drop=True)

            # we get pairs of note on and note off events for each key-press
            note_ons["end"] = note_offs["end"]

            note_ons.loc[note_ons["end"] <= note_ons["start"]] = np.nan
            note_ons = note_ons.dropna(axis=0)
            note_groups.append(note_ons)

        notes = pd.concat(note_groups, axis=0, ignore_index=True).reset_index(drop=True)

        notes["end"] = notes["end"].fillna(notes["end"].max())
        # Make all the notes that were pressed for less than min_time_unit have a duration of min_time_unit
        notes.loc[notes["end"] == notes["start"], "end"] += self.min_time_unit
        notes = notes.sort_values(by="start")
        notes = notes.reset_index(drop=True)

        return notes
