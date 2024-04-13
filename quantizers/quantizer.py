from abc import abstractmethod

import pandas as pd
from fortepyan import MidiPiece


class MidiQuantizer:
    def quantize_piece(self, piece: MidiPiece) -> MidiPiece:
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        df = self.apply_quantization(df)
        out = MidiPiece(df=df, source=source)
        return out

    def inject_quantization_features(self, piece: MidiPiece) -> MidiPiece:
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        out = MidiPiece(df=df, source=source)
        return out

    @abstractmethod
    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
