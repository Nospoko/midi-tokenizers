import re

import numpy as np
import pandas as pd

from midi_tokenizers.base_tokenizers.midi_tokenizer import MidiTokenizer


class ExponentialTimeTokenizer(MidiTokenizer):
    def __init__(
        self,
        vocab: list[str],
        first_placeholder_id: int,
        tokenizer_config: dict,
    ):
        """Initialize tokenizer with vocabulary and config.

        Args:
            vocab: All available tokens
            first_placeholder_id: Starting ID for placeholder tokens
            token_to_value: Maps tokens to semantic values (pitch/velocity/time)
            tokenizer_config: Contains min_time_unit, n_velocity_bins, n_special_ids
        """
        super().__init__(
            vocab=vocab,
            tokenizer_config=tokenizer_config,
        )
        self.first_placeholder_id = first_placeholder_id
        self.min_time_unit = self.tokenizer_config["min_time_unit"]
        self.n_velocity_bins = self.tokenizer_config["n_velocity_bins"]
        self.n_special_ids = self.tokenizer_config["n_special_ids"]

        self.velocity_bin_edges = np.linspace(
            0,
            128,
            num=self.n_velocity_bins + 1,
            endpoint=True,
        ).astype(int)

        self._build_velocity_decoder()
        self.name = "ExponentialTimeTokenizer"

        # List the tokens and step sizes describing time, to use during time tokenization
        self.time_tokens = [t for t in self.vocab if t.endswith("T")]
        self.step_sizes = sorted([ExponentialTimeTokenizer.token_to_value(t) for t in self.time_tokens], reverse=True)

    @classmethod
    def build_tokenizer(cls, tokenizer_config: dict) -> "ExponentialTimeTokenizer":
        lexicon = cls._build_lexicon(tokenizer_config=tokenizer_config)

        tokenizer = cls(
            vocab=lexicon["vocab"],
            first_placeholder_id=lexicon["first_placeholder_id"],
            tokenizer_config=tokenizer_config,
        )
        return tokenizer

    def __rich_repr__(self):
        yield "min_time_unit", self.min_time_unit
        yield "vocab_size", self.vocab_size
        yield "n_placeholder_tokens", self.n_placeholder_tokens

    @property
    def parameters(self):
        return {
            "vocab": self.vocab,
            "first_placeholder_id": self.first_placeholder_id,
            "tokenizer_config": self.tokenizer_config,
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def n_placeholder_tokens(self):
        return self.vocab_size - self.first_placeholder_id

    # Token definitions
    @staticmethod
    def pitch_to_on_token(pitch: int) -> str:
        return f"NOTE_ON_{pitch}"

    @staticmethod
    def pitch_to_off_token(pitch: int) -> str:
        return f"NOTE_OFF_{pitch}"

    @staticmethod
    def velocity_bin_to_token(bin: int) -> str:
        return f"VELOCITY_{bin}"

    @staticmethod
    def step_to_token(step: int) -> str:
        return f"{step}T"

    @staticmethod
    def token_to_value(token: str) -> int:
        """Finds a numerical value in a string and returns it"""
        match = re.search(r"\d+", token)
        return int(match.group()) if match else 0

    @classmethod
    def _build_lexicon(cls, tokenizer_config: dict) -> dict:
        """Construct tokenizer vocabulary and mappings.

        Creates tokens for:
            - (<PAD>, <CLS>)
            - MIDI notes (NOTE_ON/OFF_21-108)
            - Velocities (VELOCITY_0-N)
            - Time steps (1T-64T)
            - Placeholders

        Args:
        tokenizer_config: Contains min_time_unit, n_velocity_bins, n_special_ids

        Returns:
            Dict with vocab, first_placeholder_id, and token_to_value
        """
        vocab = ["<PAD>", "<CLS>"]

        n_velocity_bins = tokenizer_config["n_velocity_bins"]
        n_special_ids = tokenizer_config["n_special_ids"]
        min_time_unit = tokenizer_config["min_time_unit"]

        # Add MIDI note and velocity tokens to the vocabulary
        for pitch in range(21, 109):
            note_on_token = cls.pitch_to_on_token(pitch=pitch)
            note_off_token = cls.pitch_to_off_token(pitch=pitch)

            vocab.append(note_on_token)
            vocab.append(note_off_token)

        for bin in range(n_velocity_bins):
            velocity_token = cls.velocity_bin_to_token(bin)
            vocab.append(velocity_token)

        time_tokens = cls._time_vocab(min_time_unit=min_time_unit)
        vocab += time_tokens

        first_placeholder_token = len(vocab)

        for it in range(n_special_ids):
            vocab.append(f"<PLACEHOLDER_{it}>")

        lexicon = {
            "vocab": vocab,
            "first_placeholder_id": first_placeholder_token,
        }
        return lexicon

    @classmethod
    def _time_vocab(cls, min_time_unit: float) -> tuple[dict, dict]:
        """
        Generate time tokens and their mappings.

        Returns:
        tuple[dict, dict]: The time vocabulary and time_steps to token mapping.
        """
        time_tokens = []

        dt = min_time_unit
        step = 1
        # Generate time tokens with exponential distribution
        while dt < 1:
            time_token = cls.step_to_token(step=step)
            time_tokens.append(time_token)
            dt *= 2
            step *= 2
        return time_tokens

    def add_special_tokens(self, special_tokens: list[str]):
        """Add custom tokens by replacing placeholders.

        Args:
            special_tokens (list[str]): New tokens to add
        """
        new_vocab = self.vocab.copy()
        for special_token in special_tokens:
            # Switch the placeholder token for a special token
            new_vocab[self.first_placeholder_id] = special_token
            self.first_placeholder_id += 1
        self.vocab = new_vocab

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

    def _steps_to_time(self, steps: int) -> float:
        return steps * self.min_time_unit

    def tokenize_time_distance(self, dt: float) -> list[str]:
        steps = self._time_to_steps(dt)
        tokenized_time = []
        remaining_steps = steps

        for step in self.step_sizes:
            while remaining_steps >= step:
                tokenized_time.append(self.step_to_token(step))
                remaining_steps -= step

        if remaining_steps > 0:
            raise ValueError(
                f"Unable to exactly tokenize time distance: {dt} ({steps} steps), remaining: {remaining_steps} steps"
            )

        return tokenized_time

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
                tokens.append(self.velocity_bin_to_token(event["velocity_bin"]))
                tokens.append(self.pitch_to_on_token(event["pitch"]))
            else:
                tokens.append(self.pitch_to_off_token(event["pitch"]))
            previous_time = event["time"]

        return tokens

    def untokenize(self, tokens: list[str], complete_notes: bool = False) -> pd.DataFrame:
        events = []
        current_time = 0.0
        current_velocity = 0

        for token in tokens:
            if token.endswith("T"):
                # For time tokens, token_to_value holds number of steps
                current_time += self.token_to_value(token) * self.min_time_unit
            elif token.startswith("VELOCITY"):
                current_velocity = self.bin_to_velocity[self.token_to_value(token)]
            elif token.startswith("NOTE_ON"):
                events.append((current_time, "on", self.token_to_value(token), current_velocity))
            elif token.startswith("NOTE_OFF"):
                events.append((current_time, "off", self.token_to_value(token), 0))

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
