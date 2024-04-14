import numpy as np
import pandas as pd

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class NoLossTokenizer(MidiTokenizer):
    def __init__(
        self,
        eps: float = 0.001,
        n_velocity_bins: int = 128,
    ):
        super().__init__()
        self.eps = eps
        self.n_velocity_bins = n_velocity_bins
        self.specials = ["<CLS>"]
        self.vocab = self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

        self.velocity_bin_edges = np.linspace(0, 127, num=n_velocity_bins, endpoint=True).astype(int)
        self.bin_to_velocity = self._build_velocity_decoder()

    def __rich_repr__(self):
        yield "NoLossTokenizer"
        yield "eps", self.eps
        yield "vocab_size", self.vocab_size

    def _build_vocab(self):
        self.vocab = list(self.specials)

        # Add MIDI note and velocity tokens to the vocabulary
        for pitch in range(21, 109):
            self.vocab.append(f"NOTE_ON_{pitch}")
            self.vocab.append(f"NOTE_OFF_{pitch}")

        for vel in range(self.n_velocity_bins):
            self.vocab.append(f"VELOCITY_{vel}")

        token = self.eps

        # Generate time tokens with exponential distribution
        while token < 1:
            self.vocab.append(f"{token}s")
            token *= 2

        self.max_time_token = token  # Maximum time token
        return self.vocab

    def quantize_frame(self, df: pd.DataFrame):
        df["velocity_bin"] = np.digitize(df.velocity, self.velocity_bin_edges) - 1
        return df

    def _build_velocity_decoder(self):
        self.bin_to_velocity = []
        for it in range(1, len(self.velocity_bin_edges)):
            velocity = (self.velocity_bin_edges[it - 1] + self.velocity_bin_edges[it]) / 2
            self.bin_to_velocity.append(int(velocity))
        return self.bin_to_velocity

    @staticmethod
    def _notes_to_events(notes: pd.DataFrame):
        """
        Convert MIDI note dataframe into a dict with on/off events.
        """
        note_on_df: pd.DataFrame = notes.loc[:, ["start", "pitch", "velocity_bin"]]
        note_off_df: pd.DataFrame = notes.loc[:, ["end", "pitch"]]

        note_off_df["time"] = note_off_df["end"]
        note_off_df["event"] = "NOTE_OFF"
        note_on_df["time"] = note_on_df["start"]
        note_on_df["event"] = "NOTE_ON"

        note_on_events = note_on_df.to_dict(orient="records")
        note_off_events = note_off_df.to_dict(orient="records")
        note_events = note_on_events + note_off_events

        note_events = sorted(note_events, key=lambda event: event["time"])
        return note_events

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        notes = self.quantize_frame(notes)
        tokens = []
        # Time difference between current and previous events
        previous_time = 0
        note_events = self._notes_to_events(notes=notes)

        for current_event in note_events:
            # Check timing beginning with the largest step
            current_step = self.max_time_token
            # Calculate the time difference between current and previous event
            dt = current_event["time"] - previous_time
            filling_dt = 0  # filled time difference
            # Fill the time gap
            while True:
                if filling_dt + current_step > dt:
                    # Select time step that will fit into the gap
                    current_step /= 2
                else:
                    # Fill the gap with current time token
                    tokens.append(f"{current_step}s")
                    filling_dt += current_step

                if dt - filling_dt < self.eps:
                    # Exit the loop when the gap is filled
                    break

            event_type = current_event["event"]
            # Append note event tokens
            if event_type == "NOTE_ON":
                velocity = int(current_event["velocity_bin"])
                tokens.append(f"VELOCITY_{velocity}")

            pitch = int(current_event["pitch"])
            tokens.append(f"{event_type}_{pitch}")

            previous_time = current_event["time"]

        return tokens

    def untokenize(self, tokens: list[str]) -> pd.DataFrame:
        note_on_events = []
        note_off_events = []

        current_time = 0
        current_velocity = 0
        for token in tokens:
            if "s" in token:
                dt: float = eval(token[:-1])
                current_time += dt
            if "VELOCITY" in token:
                # velocity should always be right before NOTE_ON token
                current_velocity_bin: int = eval(token[9:])
                current_velocity: int = self.bin_to_velocity[current_velocity_bin]
            if "NOTE_ON" in token:
                note = {
                    "pitch": eval(token[8:]),
                    "start": current_time,
                    "velocity": current_velocity,
                }
                note_on_events.append(note)
            if "NOTE_OFF" in token:
                note = {
                    "pitch": eval(token[9:]),
                    "end": current_time,
                }
                note_off_events.append(note)

        # Both should be sorted by time right now
        note_on_events = pd.DataFrame(note_on_events)
        note_off_events = pd.DataFrame(note_off_events)

        # So if we sort them by pitch ...
        note_on_events = note_on_events.sort_values(by="pitch", kind="stable").reset_index(drop=True)
        note_off_events = note_off_events.sort_values(by="pitch", kind="stable").reset_index(drop=True)

        # we get pairs of note on and note off events for each key-press
        notes = note_on_events
        notes["end"] = note_off_events["end"]

        notes = notes.sort_values(by="start")
        notes = notes.reset_index(drop=True)

        return notes
