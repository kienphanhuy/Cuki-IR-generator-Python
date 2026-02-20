#!/usr/bin/env -S uv run
# -*- coding: utf-8 -*-
"""
Cuki IR Generator
=================
Generates acoustic guitar IR (Impulse Response) files from a stereo wav recording.

The input wav file must be stereo:
  - Left channel  : piezo pickup (pic)
  - Right channel : microphone (mic)

Output files (written to ./output/ by default):
  IR_<name>_<fs>k_2048_M.wav        – Modified process IR (usually clearer)
  IR_<name>_<fs>k_2048_Std.wav      – Standard IR
  IR_<name>_<fs>k_2048_Std_Bld.wav  – Standard IR blended 50/50 with dry pickup
  IR_<name>_<fs>k_2048_Std.png      – Std IR waveform + spectrum plot
  IR_<name>_<fs>k_2048_M.png        – Modified IR waveform + spectrum plot

Usage:
  uv run cuki_ir_cli.py <input.wav> [--ir-size 2048] [--output-dir ./output]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from scipy import signal


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate acoustic guitar IR files from a stereo mic/pickup WAV recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="INPUT_WAV",
        help="Path to the stereo input WAV file (left=pickup, right=mic).",
    )
    parser.add_argument(
        "--ir-size",
        type=int,
        default=2048,
        metavar="N",
        help="IR length in samples (power of 2 recommended, e.g. 512/1024/2048).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./output"),
        metavar="DIR",
        help="Directory where output WAV files and graphs are saved.",
    )
    return parser.parse_args()


# ── DSP helpers ────────────────────────────────────────────────────────────────

def oct_spectrum2(s, fs):
    """Compute 1/3-octave band spectrum of signal s sampled at fs."""
    b = 3
    dbref = 1
    G = 10 ** (3 / 10)
    fr = 1000
    x = np.arange(1, 43)
    Gstar = np.power(G, (x - 30) / b)
    fm = np.multiply(Gstar, fr)
    f2 = np.multiply(np.power(G, 1 / (2 * b)), fm)
    x  = np.delete(x,  np.where(f2 < 20),   axis=0)
    x  = np.delete(x,  np.where(f2 > fs/2), axis=0)
    fm = np.delete(fm, np.where(f2 < 20),   axis=0)
    fm = np.delete(fm, np.where(f2 > fs/2), axis=0)
    f2 = np.delete(f2, np.where(f2 < 20),   axis=0)
    f2 = np.delete(f2, np.where(f2 > fs/2), axis=0)
    f1 = np.multiply(np.power(G, -1 / (2 * b)), fm)

    S = np.zeros(len(x))
    for k in range(len(x)):
        B, A = signal.butter(2, [f1[k], f2[k]], btype='bandpass', fs=fs)
        sfilt = signal.lfilter(B, A, s)
        rms2b = np.sqrt(1 / len(sfilt) * sum(sfilt ** 2))
        S[k] = 10 * np.log10((rms2b / dbref) ** 2)

    rms2 = np.sqrt(1 / len(s) * sum(np.power(s, 2)))
    overall_lev  = 10 * np.log10(np.power(np.divide(rms2, dbref), 2))
    overall_levA = 10 * np.log10(np.sum(np.power(10, S / 10)))
    return S, fm, overall_lev, overall_levA, f1, f2


def save_ir_plot(ir, fs, NbF, title_waveform, title_spectrum, out_path):
    """Save a two-panel figure: IR waveform + log-frequency spectrum."""
    from matplotlib import pyplot as plt
    t    = np.arange(0, len(ir)) / fs
    FIRX = np.fft.fft(ir, NbF)
    freq = np.fft.fftfreq(len(t), t[1] - t[0])
    SdB  = 20 * np.log10(np.maximum(np.absolute(FIRX), 1e-12))  # avoid log(0)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(t * 1000, ir)
    axs[0].set_title(title_waveform)
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()

    axs[1].plot(freq[0:int(NbF / 2)], SdB[0:int(NbF / 2)])
    axs[1].set_title(title_spectrum)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("dB")
    axs[1].grid()
    axs[1].set_xscale('log')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  graph saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    NbF        = args.ir_size
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a clean stem for output filenames (no extension)
    stem = input_path.stem  # e.g. "RmicLpic"

    # ── Load audio ─────────────────────────────────────────────────────────────
    print(f"Loading {input_path} …")
    data, fs = sf.read(input_path)
    if data.ndim < 2:
        print("Error: input file must be stereo (left=pickup, right=mic).", file=sys.stderr)
        sys.exit(1)

    Nb   = data.shape[0]
    mic  = data[:, 1]   # right channel = microphone
    pic  = data[:, 0]   # left  channel = piezo pickup

    # ── Build frame-by-frame FIR estimates ────────────────────────────────────
    print(f"Computing IR (NbF={NbF}) …")
    FIR1  = np.zeros(NbF, dtype=complex)
    Nbuff = fs
    Nbmax = math.floor(Nb / Nbuff) - 10
    alice = np.zeros((NbF, Nbmax), dtype=complex)

    for n in range(Nbmax):
        i   = 3 * fs + n * Nbuff
        FIR = np.divide(
            np.fft.fft(mic[i:i + Nbuff - 1], NbF),
            np.fft.fft(pic[i:i + Nbuff - 1], NbF),
        )
        IR = np.real(np.fft.ifft(FIR, NbF))
        IR = IR / np.amax(np.absolute(IR))
        if any(np.isinf(FIR)) or any(np.isnan(FIR)):
            IR  = np.zeros(Nbuff)
            IR[0] = 1
            FIR = np.fft.fft(IR, NbF)
            print("  Warning: NaN or Inf detected in frame", n)
        alice[0:NbF, n] = FIR
        FIR1 = FIR1 + FIR

    # ── Average across frames (reject outliers beyond 2 std-devs) ─────────────
    ALICE = np.zeros(NbF, dtype=complex)
    for i in range(NbF):
        a = alice[i, :]
        A = a[np.absolute(np.absolute(a) - np.mean(a)) < 2 * np.std(a)]
        if any(np.isnan(A)) or any(np.isinf(A)):
            A = 1
            print(f"  Warning: NaN/Inf at frequency bin {i}")
        else:
            A = np.mean(A)
        ALICE[i] = A

    # ── Apply Blackman window & normalise ─────────────────────────────────────
    nn2       = np.arange(0, int(2 * NbF))
    window    = (.42 - .5 * np.cos(2 * np.pi * nn2 / (2 * NbF - 1))
                     + .08 * np.cos(4 * np.pi * nn2 / (2 * NbF - 1)))
    blackmanwin = window[NbF - 1:len(window) - 1]

    ir2  = np.fft.ifft(ALICE)
    ir2  = np.multiply(ir2, blackmanwin)
    ir2  = ir2 / np.amax(np.absolute(ir2)) * 0.95
    IR2  = np.real(ir2)   # Standard IR

    # ── Equalisation correction (Modified IR) ─────────────────────────────────
    nn3 = np.arange(10 * fs + 1, 20 * fs)
    MS  = mic[nn3]
    PS  = np.convolve(pic[nn3], IR2, 'same')
    p,  cf, _, _, f1, f2 = oct_spectrum2(MS / np.amax(np.absolute(MS)), fs)
    p2, _,  _, _, _,  _  = oct_spectrum2(PS / np.amax(np.absolute(PS)), fs)
    g0 = p - p2

    IRX = np.zeros(NbF); IRX[0] = 1
    IR1 = IR2.copy()
    for i in range(len(f1)):
        g  = 10 ** ((g0[i]) / 20)
        B, A = signal.butter(2, [f1[i], f2[i]], btype='bandpass', fs=fs)
        sfilt  = signal.lfilter(B, A, IRX) * (g - 1)
        IRX    = IRX + sfilt
        sfilt1 = signal.lfilter(B, A, IR1) * (g - 1)
        IR1    = IR1 + sfilt1

    IR1 = IR1 / np.amax(np.absolute(IR1)) * 0.95   # Modified IR

    # ── Blended IR (50% dry pickup + 50% Standard IR) ─────────────────────────
    IRX = np.zeros(NbF); IRX[0] = 1
    IR3 = (IRX + IR2) / 2

    # ── Output filename prefix ─────────────────────────────────────────────────
    fmt = str(fs / 1000)
    prefix = f"IR_{stem}_{fmt[0:2]}k_{NbF}"

    # ── Save WAV files ─────────────────────────────────────────────────────────
    for ir, suffix in [(IR1, "_M"), (IR2, "_Std"), (IR3, "_Std_Bld")]:
        out = output_dir / f"{prefix}{suffix}.wav"
        print(f"Saving {out} …")
        sf.write(str(out), ir, fs, 'PCM_24')

    # ── Save plots ─────────────────────────────────────────────────────────────
    print("Generating plots …")
    spec_dir = output_dir / "spectrum_graphs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    save_ir_plot(
        IR2, fs, NbF,
        title_waveform="IR Std",
        title_spectrum="Std Spectrum",
        out_path=spec_dir / f"{prefix}_Std.png",
    )
    save_ir_plot(
        IR1, fs, NbF,
        title_waveform="IR M-file",
        title_spectrum="M-file Spectrum",
        out_path=spec_dir / f"{prefix}_M.png",
    )

    print("\n✅ Done!")
    print("  _Std      : Standard algorithm")
    print("  _Std_Bld  : Standard with 50% raw pickup / 50% IR blend")
    print("  _M        : Modified process (usually clearer than the standard)")


if __name__ == "__main__":
    main()
