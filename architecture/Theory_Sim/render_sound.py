"""
render_sound.py

FM renderer for contact dynamics delta waveforms.

Four commensurable signals only — contact rate, expansion-ready,
free space, contact flux. All boundary state transition rates,
all fractional quantities, all telling the same story from
complementary angles.

Mean gap excluded — different physical quantity (spatial extent),
different shape, different story. Kept in CSV for reference.

Each signal normalised to its own peak, then layered in the mix.
Single final level normalisation only — no content change.

Output: theory_sim_1/sound_data/contact_dynamics.wav
"""

import numpy as np
import scipy.io.wavfile as wav
from scipy.interpolate import interp1d
import os

SOUND_FOLDER   = "theory_sim_L1_delta_wav/sound_data"
OUTPUT_WAV     = os.path.join(SOUND_FOLDER, "contact_dynamics.wav")
SAMPLE_RATE    = 44100
DURATION_SEC   = 20
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SEC

# Four boundary state signals — commensurable fractional quantities
CARRIERS = [220.0, 330.0, 440.0, 550.0]
FM_DEPTH = 600.0
PANS     = [-0.6, -0.2, 0.2, 0.6]

SIGNAL_FILES = [
    "delta_contact_rate.csv",
    "delta_expansion_ready.csv",
    "delta_free_space.csv",
    "delta_contact_flux.csv",
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
    inst_freq    = np.clip(carrier_hz + fm_depth * delta_norm, 20.0, 20000.0)
    phase        = np.cumsum(2.0 * np.pi * inst_freq / sample_rate)
    carrier_wave = np.sin(phase)
    amplitude    = np.abs(delta_norm)
    return carrier_wave * amplitude

def pan_to_stereo(mono, pan):
    angle = (pan + 1.0) / 2.0 * (np.pi / 2.0)
    return np.cos(angle) * mono, np.sin(angle) * mono

def run():
    n         = TARGET_SAMPLES
    left_mix  = np.zeros(n)
    right_mix = np.zeros(n)

    for i, fname in enumerate(SIGNAL_FILES):
        path         = os.path.join(SOUND_FOLDER, fname)
        gens, raw    = load_delta(path)
        interpolated = interpolate_to_samples(gens, raw, n)
        delta_norm   = normalise_to_own_peak(interpolated)
        voice        = render_voice(CARRIERS[i], delta_norm, FM_DEPTH, SAMPLE_RATE)
        l, r         = pan_to_stereo(voice, PANS[i])
        left_mix    += l
        right_mix   += r
        rms = np.sqrt(np.mean(voice ** 2))
        print(f"  {fname:<35}  carrier={CARRIERS[i]:5.0f}Hz  pan={PANS[i]:+.1f}  rms={rms:.4f}")

    peak = max(np.abs(left_mix).max(), np.abs(right_mix).max())
    if peak > 0:
        left_mix  /= peak
        right_mix /= peak

    stereo       = np.stack([left_mix, right_mix], axis=1)
    stereo_int16 = (stereo * 32767).astype(np.int16)
    wav.write(OUTPUT_WAV, SAMPLE_RATE, stereo_int16)
    print(f"\n  Written: {OUTPUT_WAV}  |  {DURATION_SEC}s  |  {SAMPLE_RATE}Hz  |  Stereo")

if __name__ == "__main__":
    run()