from abc import abstractmethod

import pandas as pd
from fortepyan import MidiPiece


class MidiQuantizer:
    """
    Base class for MIDI quantizers.

    Provides methods to quantize a MIDI piece and inject quantization features.
    Subclasses must implement the `apply_quantization` and `quantize_frame` methods.

    Methods:
        quantize_piece(piece: MidiPiece) -> MidiPiece:
            Quantizes a given MIDI piece and returns a new MidiPiece object.

        inject_quantization_features(piece: MidiPiece) -> MidiPiece:
            Injects quantization features into a given MIDI piece and returns a new MidiPiece object.

    Abstract Methods:
        apply_quantization(df: pd.DataFrame) -> pd.DataFrame:
            Applies the quantization to the DataFrame. Must be implemented by subclasses.

        quantize_frame(df: pd.DataFrame) -> pd.DataFrame:
            Quantizes the frame of the DataFrame. Must be implemented by subclasses.
    """

    def quantize_piece(self, piece: MidiPiece) -> MidiPiece:
        """
        Quantizes a given MIDI piece and returns a new MidiPiece object.

        Parameters:
        piece (MidiPiece): The MIDI piece to quantize.

        Returns:
        MidiPiece: A new MidiPiece object with quantized data.
        """
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        df = self.apply_quantization(df)
        out = MidiPiece(df=df, source=source)
        return out

    def inject_quantization_features(self, piece: MidiPiece) -> MidiPiece:
        """
        Injects quantization features into a given MIDI piece and returns a new MidiPiece object.

        Parameters:
        piece (MidiPiece): The MIDI piece to inject quantization features into.

        Returns:
        MidiPiece: A new MidiPiece object with quantization features injected.
        """
        # Try not to overwrite anything
        df = piece.df.copy()
        source = dict(piece.source) | {"quantized": True}

        # Make the quantization
        df = self.quantize_frame(df)
        out = MidiPiece(df=df, source=source)
        return out

    @abstractmethod
    def apply_quantization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the quantization to the DataFrame.

        Must be implemented by subclasses.

        Parameters:
        df (pd.DataFrame): The DataFrame to apply quantization to.

        Returns:
        pd.DataFrame: The DataFrame with quantization applied.
        """
        pass

    @abstractmethod
    def quantize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Quantizes the frame of the DataFrame.

        Must be implemented by subclasses.

        Parameters:
        df (pd.DataFrame): The DataFrame to quantize.

        Returns:
        pd.DataFrame: The quantized DataFrame.
        """
        pass
