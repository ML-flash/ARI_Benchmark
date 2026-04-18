"""
render_sound_L6.py

FM renderer for Layer 6 delta waveforms.

Two signal groups, two stereo WAV files, one combined mix:

  Group A — Boundary Dynamics (4 voices):
    Δ contact rate, Δ expansion-ready, Δ free space, Δ contact flux
    Same carriers and panning as L1 for continuity.

  Group B — Fitness/Selection Dynamics (4 voices):
    Δ mean fitness, Δ mean decoded length,
    Δ fraction over threshold, Δ service set size

  Combined — All 8 voices in one stereo field.

Each signal normalised to its own peak, then layered in the mix.
Single final level normalisation only — no content change.

Output:
  theory_sim_L6_delta_wav/sound_data/
    boundary_dynamics.wav       (Group A only)
    fitness_dynamics.wav        (Group B only)
    combined_dynamics.wav       (All 8 signals)
"""

import numpy as np
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
import os
import sys

SOUND_FOLDER   = "theory_sim_L6_delta_wav/sound_data"
SAMPLE_RATE    = 44100
DURATION_SEC   = 30       # Longer than L1 — 5000 gens vs 600
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SEC

FM_DEPTH = 600.0

# Group A: Boundary dynamics — same carriers as L1
BOUNDARY_CARRIERS = [220.0, 330.0, 440.0, 550.0]
BOUNDARY_PANS     = [-0.6, -0.2, 0.2, 0.6]
BOUNDARY_FILES    = [
    "delta_contact_rate.csv",
    "delta_expansion_ready.csv",
    "delta_free_space.csv",
    "delta_contact_flux.csv",
]

# Group B: Fitness/selection dynamics — distinct carrier register
FITNESS_CARRIERS = [165.0, 277.0, 370.0, 493.0]
FITNESS_PANS     = [-0.5, -0.15, 0.15, 0.5]
FITNESS_FILES    = [
    "delta_mean_fitness.csv",
    "delta_mean_decoded_len.csv",
    "delta_frac_over_threshold.csv",
    "delta_service_set_size.csv",
]


def load_delta(filename):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def interpolate_to_samples(gens, signal, n_samples):
    x_new = np.linspace(gens[0], gens[-1], n_samples)
    return interp1d(gens, signal, kind="cubic")(x_new)


def normalise_to_own_peak(signal):
    peak = np.abs(signal).max()
    return signal / peak if peak > 0 else signal


def render_voice(carrier_hz, delta_norm, fm_depth, sample_rate):
    """FM synthesis: freq = carrier + fm_depth * delta, amplitude = |delta|."""
    inst_freq    = np.clip(carrier_hz + fm_depth * delta_norm, 20.0, 20000.0)
    phase        = np.cumsum(2.0 * np.pi * inst_freq / sample_rate)
    carrier_wave = np.sin(phase)
    amplitude    = np.abs(delta_norm)
    return carrier_wave * amplitude


def pan_to_stereo(mono, pan):
    """Equal-power panning. pan in [-1, 1]."""
    angle = (pan + 1.0) / 2.0 * (np.pi / 2.0)
    return np.cos(angle) * mono, np.sin(angle) * mono


def render_group(signal_files, carriers, pans):
    """Render a group of signals to stereo mix. Returns (left, right) arrays."""
    n = TARGET_SAMPLES
    left_mix  = np.zeros(n)
    right_mix = np.zeros(n)

    for i, fname in enumerate(signal_files):
        path = os.path.join(SOUND_FOLDER, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        gens, raw    = load_delta(path)
        interpolated = interpolate_to_samples(gens, raw, n)
        delta_norm   = normalise_to_own_peak(interpolated)
        voice        = render_voice(carriers[i], delta_norm, FM_DEPTH, SAMPLE_RATE)
        l, r         = pan_to_stereo(voice, pans[i])
        left_mix    += l
        right_mix   += r

        rms = np.sqrt(np.mean(voice ** 2))
        print(f"  {fname:<40}  carrier={carriers[i]:5.0f}Hz  "
              f"pan={pans[i]:+.2f}  rms={rms:.4f}")

    return left_mix, right_mix


def normalise_and_write(left, right, output_path):
    """Final level normalisation and WAV write."""
    peak = max(np.abs(left).max(), np.abs(right).max())
    if peak > 0:
        left  = left  / peak
        right = right / peak

    stereo       = np.stack([left, right], axis=1)
    stereo_int16 = (stereo * 32767).astype(np.int16)
    wav.write(output_path, SAMPLE_RATE, stereo_int16)
    print(f"  Written: {output_path}")


def run():
    print(f"\n  Layer 6 Sound Render")
    print(f"  Duration: {DURATION_SEC}s  |  {SAMPLE_RATE}Hz  |  Stereo\n")

    # Check that the sim has been run
    if not os.path.isdir(SOUND_FOLDER):
        print(f"  ERROR: {SOUND_FOLDER} does not exist.")
        print(f"  Run theory_sim_L6_delta_wav.py first.")
        return

    test_file = os.path.join(SOUND_FOLDER, BOUNDARY_FILES[0])
    if not os.path.exists(test_file):
        print(f"  ERROR: {test_file} not found.")
        print(f"  Run theory_sim_L6_delta_wav.py first to generate the delta CSVs.")
        return

    # Group A: Boundary dynamics
    print("  --- Group A: Boundary Dynamics ---")
    bl, br = render_group(BOUNDARY_FILES, BOUNDARY_CARRIERS, BOUNDARY_PANS)
    normalise_and_write(bl, br, os.path.join(SOUND_FOLDER, "boundary_dynamics.wav"))

    # Group B: Fitness/selection dynamics
    print("\n  --- Group B: Fitness/Selection Dynamics ---")
    fl, fr = render_group(FITNESS_FILES, FITNESS_CARRIERS, FITNESS_PANS)
    normalise_and_write(fl, fr, os.path.join(SOUND_FOLDER, "fitness_dynamics.wav"))

    # Combined: all 8 voices
    print("\n  --- Combined: All Dynamics ---")
    combined_left  = bl + fl
    combined_right = br + fr
    normalise_and_write(combined_left, combined_right,
                        os.path.join(SOUND_FOLDER, "combined_dynamics.wav"))

    print(f"\n  Three WAV files produced:")
    print(f"    boundary_dynamics.wav  — boundary geometry only (L0 dynamics)")
    print(f"    fitness_dynamics.wav   — fitness/selection only (L6 dynamics)")
    print(f"    combined_dynamics.wav  — all 8 signals (full feedback loop)")


if __name__ == "__main__":
    run()