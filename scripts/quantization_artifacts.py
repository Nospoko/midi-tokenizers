import yaml
import numpy as np
import pandas as pd
from fortepyan import MidiPiece
from datasets import load_dataset


def create_quantization_artifacts():
    dataset = load_dataset("roszcz/maestro-v1-sustain", split="train")
    pieces = [MidiPiece.from_huggingface(hf) for hf in dataset]

    n_dstarts = range(3, 11)
    n_durations = range(3, 11)
    n_velocities = range(1, 11)

    durations = {}
    for n_duration in n_durations:
        duration_bin_edges = find_dataset_duration_bin_edges(pieces, n_duration)
        durations[n_duration] = duration_bin_edges.tolist()

    dstarts = {}
    for n_dstart in n_dstarts:
        dstart_bin_edges = find_dataset_dstart_bin_edges(pieces, n_dstart)
        dstarts[n_dstart] = dstart_bin_edges.tolist()

    velocities = {}
    for n_velocity in n_velocities:
        velocity_bin_edges = find_dataset_velocity_bin_edges(pieces, n_velocity)
        velocities[n_velocity] = velocity_bin_edges.tolist()

    bin_edges = {
        "velocity": velocities,
        "duration": durations,
        "dstart": dstarts,
    }

    with open("midi_quantization_artifacts/bin_edges.yaml", "w") as f:
        yaml.dump(bin_edges, f)

    return bin_edges


def find_dataset_dstart_bin_edges(pieces: list[MidiPiece], n_bins: int = 3) -> np.array:
    dstarts = []
    for piece in pieces:
        next_start = piece.df.start.shift(-1)
        dstart = next_start - piece.df.start
        # Last value is nan
        dstarts.append(dstart[:-1])

    # We're not doing num=n_bins + 1 here (like in other functions)
    # Because the last edge is handcraftet ...
    quantiles = np.linspace(0, 1, num=n_bins)

    dstarts = np.hstack(dstarts)
    bin_edges = np.quantile(dstarts, quantiles)[:-1]

    bin_edges[0] = 0
    # ... here:
    # dstart is mostly distributed in low values, but
    # we need to have at least one token for longer notes
    last_edge = max(bin_edges[-1] * 3, 0.5)
    bin_edges = np.append(bin_edges, last_edge)
    return bin_edges


def find_dataset_duration_bin_edges(pieces: list[MidiPiece], n_bins: int = 3) -> np.array:
    df = pd.concat([p.df for p in pieces])
    duration = df.duration.values

    quantiles = np.linspace(0, 1, num=n_bins + 1)

    bin_edges = np.quantile(duration, quantiles)[:-1]

    bin_edges[0] = 0
    return bin_edges


def find_dataset_velocity_bin_edges(pieces: list[MidiPiece], n_bins: int = 3) -> np.array:
    velocities = np.hstack([p.df.velocity.values for p in pieces])

    quantiles = np.linspace(0, 1, num=n_bins + 1)

    bin_edges = np.quantile(velocities, quantiles)

    bin_edges[0] = 0
    bin_edges[-1] = 128
    return bin_edges
