import pandas as pd

from midi_tokenizers.midi_tokenizer import MidiTokenizer


class NoLossTokenizer(MidiTokenizer):
    def __init__(
        self,
        eps: float = 0.001,
    ):
        super().__init__()
        self.eps = eps
        self.specials = ["<CLS>"]
        self.vocab = self._build_vocab()
        self.token_to_id = {token: it for it, token in enumerate(self.vocab)}

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

        for vel in range(128):
            self.vocab.append(f"VELOCITY_{vel}")

        token = self.eps

        # Generate time tokens with exponential distribution
        while token < 1:
            self.vocab.append(f"{token}s")
            token *= 2

        self.max_time_token = token  # Maximum time token
        return self.vocab

    @staticmethod
    def _notes_to_event_df(notes: pd.DataFrame):
        """
        Convert MIDI note dataframe into a dataframe with on/off events.
        """
        note_on_events: pd.DataFrame = notes.loc[:, ["start", "pitch", "velocity"]]
        note_off_events: pd.DataFrame = notes.loc[:, ["end", "pitch"]]

        note_off_events["time"] = note_off_events["end"]
        note_off_events["event"] = "NOTE_OFF"
        note_on_events["time"] = note_on_events["start"]
        note_on_events["event"] = "NOTE_ON"

        note_events: pd.DataFrame = pd.concat([note_on_events, note_off_events], axis=0)
        note_events = note_events.sort_values(by="time")

        return note_events

    def tokenize(self, notes: pd.DataFrame) -> list[str]:
        tokens = []
        # Time difference between current and previous events
        previous_time = 0
        note_events = self._notes_to_event_df(notes=notes)

        for _, current_event in note_events.iterrows():
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

            # Append note event tokens
            if current_event["event"] == "NOTE_ON":
                tokens.append(f"VELOCITY_{current_event['velocity']}")
            tokens.append(f"{current_event['event']}_{current_event['pitch']}")

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
                current_velocity: int = eval(token[9:])
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
