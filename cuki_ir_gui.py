#!/usr/bin/env python3
"""
Cuki IR Generator — Graphical Frontend
=======================================
A simple tkinter UI that wraps cuki_ir.main().

Run with:
    uv run cuki_ir_gui.py
or, after pip install:
    cuki-ir-generator-gui
"""

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

import matplotlib
matplotlib.use('Agg')  # must be set before pyplot is imported anywhere


# ── redirect stdout/stderr into the GUI log ───────────────────────────────────

class _QueueWriter:
    """File-like object that puts lines into a queue for the UI thread."""
    def __init__(self, q: queue.Queue):
        self._q = q

    def write(self, text: str):
        if text:
            self._q.put(text)

    def flush(self):
        pass


# ── main window ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cuki IR Generator")
        self.resizable(True, True)
        self.minsize(560, 480)

        self._log_queue: queue.Queue = queue.Queue()
        self._running = False

        self._build_ui()
        self._poll_log()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        # ── header ────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg="#1a1a2e")
        header.pack(fill="x")
        tk.Label(
            header,
            text="Cuki IR Generator",
            font=("Helvetica", 16, "bold"),
            fg="#e0e0ff",
            bg="#1a1a2e",
            pady=12,
        ).pack()

        # ── form ──────────────────────────────────────────────────────────────
        form = tk.Frame(self, padx=16, pady=10)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        # Input WAV
        tk.Label(form, text="Input WAV (Stereo, L: pickup, R: mic):", anchor="w").grid(
            row=0, column=0, sticky="w", **pad
        )
        self._input_var = tk.StringVar()
        tk.Entry(form, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", **pad
        )
        tk.Button(form, text="Browse…", command=self._browse_input).grid(
            row=0, column=2, **pad
        )

        # Output dir
        tk.Label(form, text="Output directory:", anchor="w").grid(
            row=1, column=0, sticky="w", **pad
        )
        self._output_var = tk.StringVar(value="./output")
        tk.Entry(form, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", **pad
        )
        tk.Button(form, text="Browse…", command=self._browse_output).grid(
            row=1, column=2, **pad
        )

        # IR size
        tk.Label(form, text="IR size (samples):", anchor="w").grid(
            row=2, column=0, sticky="w", **pad
        )
        self._irsize_var = tk.StringVar(value="2048")
        ir_combo = ttk.Combobox(
            form,
            textvariable=self._irsize_var,
            values=["512", "1024", "2048", "4096"],
            width=10,
            state="normal",
        )
        ir_combo.grid(row=2, column=1, sticky="w", **pad)

        # Separator
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10)

        # ── run button ────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, pady=6)
        btn_frame.pack()
        self._run_btn = tk.Label(
            btn_frame,
            text="▶  Generate IR",
            font=("Helvetica", 12, "bold"),
            bg="#2d7a3a",
            fg="white",
            padx=20,
            pady=8,
            cursor="hand2",
        )
        self._run_btn.pack()
        self._run_btn.bind("<Button-1>", lambda e: self._run())
        self._run_btn.bind("<Enter>",    lambda e: self._on_btn_enter())
        self._run_btn.bind("<Leave>",    lambda e: self._on_btn_leave())

        # ── log ───────────────────────────────────────────────────────────────
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=10)
        tk.Label(self, text="Log output:", anchor="w").pack(
            fill="x", padx=16, pady=(6, 0)
        )
        self._log = scrolledtext.ScrolledText(
            self,
            height=12,
            state="disabled",
            bg="#0d0d1a",
            fg="#c8c8ff",
            font=("Courier", 10),
            relief="flat",
            padx=6,
            pady=6,
        )
        self._log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Tag colours for stderr (errors shown in red)
        self._log.tag_configure("err", foreground="#ff6b6b")

    # ── button hover helpers ──────────────────────────────────────────────────

    def _on_btn_enter(self):
        if not self._running:
            self._run_btn.configure(bg="#3a9e4b")

    def _on_btn_leave(self):
        if not self._running:
            self._run_btn.configure(bg="#2d7a3a")

    # ── file dialogs ──────────────────────────────────────────────────────────

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select stereo WAV recording",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self._input_var.set(path)
            # Auto-suggest output dir next to the input file
            if self._output_var.get() == "./output":
                self._output_var.set(str(Path(path).parent / "output"))

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self._output_var.set(path)

    # ── run ───────────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return

        input_path = self._input_var.get().strip()
        if not input_path:
            self._append_log("⚠  Please select an input WAV file.\n", error=True)
            return

        try:
            ir_size = int(self._irsize_var.get().strip())
        except ValueError:
            self._append_log("⚠  IR size must be an integer.\n", error=True)
            return

        output_dir = Path(self._output_var.get().strip() or "./output")

        self._set_running(True)
        self._clear_log()

        thread = threading.Thread(
            target=self._worker,
            args=(input_path, ir_size, output_dir),
            daemon=True,
        )
        thread.start()

    def _worker(self, input_path: str, ir_size: int, output_dir: Path):
        """Run IR generation in a background thread, capturing all output."""
        # Patch sys.stdout/stderr → queue so the UI thread can display them.
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _QueueWriter(self._log_queue)
        sys.stderr = _QueueWriter(self._log_queue)
        try:
            # Import here so the GUI can open even if cuki_ir isn't on PATH
            import cuki_ir_cli
            import types

            # Fake argparse result so we don't call parse_args() (which reads sys.argv)
            args = types.SimpleNamespace(
                input=input_path,
                ir_size=ir_size,
                output_dir=output_dir,
            )

            # Inline the body of main() with our args object
            # (We call the module-level functions directly to stay DRY)
            _run_generation(args)

        except SystemExit:
            pass
        except Exception as exc:
            self._log_queue.put(f"\n❌  Error: {exc}\n")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            self._log_queue.put(None)  # sentinel → generation finished

    # ── log helpers ───────────────────────────────────────────────────────────

    def _poll_log(self):
        """Called repeatedly from the Tk event loop to drain the log queue."""
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item is None:          # sentinel: worker finished
                    self._set_running(False)
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _append_log(self, text: str, error: bool = False):
        self._log.configure(state="normal")
        self._log.insert("end", text, "err" if error else "")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _set_running(self, running: bool):
        self._running = running
        if running:
            self._run_btn.configure(
                text="⏳  Running…",
                bg="#888888",
                fg="#333333",
            )
            self._run_btn.unbind("<Button-1>")
        else:
            self._run_btn.configure(
                text="▶  Generate IR",
                bg="#2d7a3a",
                fg="white",
            )
            self._run_btn.bind("<Button-1>", lambda e: self._run())


# ── generation logic (mirrors main() but takes a namespace instead of CLI args)

def _run_generation(args):
    """Duplicate-free: import and call the real processing code from cuki_ir."""
    import math
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    from scipy import signal

    import cuki_ir_cli  # for oct_spectrum2 and save_ir_plot

    input_path  = Path(args.input)
    NbF         = args.ir_size
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem        = input_path.stem

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"Loading {input_path} …")
    data, fs = sf.read(input_path)
    if data.ndim < 2:
        raise ValueError("Input file must be stereo (left=pickup, right=mic).")

    Nb  = data.shape[0]
    mic = data[:, 1]
    pic = data[:, 0]

    # ── FIR estimation ────────────────────────────────────────────────────────
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
            IR    = np.zeros(Nbuff); IR[0] = 1
            FIR   = np.fft.fft(IR, NbF)
            print(f"  Warning: NaN/Inf in frame {n}")
        alice[:, n] = FIR
        FIR1 = FIR1 + FIR

    # ── average / outlier rejection ───────────────────────────────────────────
    ALICE = np.zeros(NbF, dtype=complex)
    for i in range(NbF):
        a = alice[i, :]
        A = a[np.absolute(np.absolute(a) - np.mean(a)) < 2 * np.std(a)]
        ALICE[i] = 1 if (any(np.isnan(A)) or any(np.isinf(A))) else np.mean(A)

    # ── Blackman window + normalise ───────────────────────────────────────────
    nn2       = np.arange(0, int(2 * NbF))
    window    = (.42 - .5  * np.cos(2 * np.pi * nn2 / (2 * NbF - 1))
                     + .08 * np.cos(4 * np.pi * nn2 / (2 * NbF - 1)))
    blackmanwin = window[NbF - 1:len(window) - 1]
    ir2  = np.real(np.multiply(np.fft.ifft(ALICE), blackmanwin))
    IR2  = ir2 / np.amax(np.absolute(ir2)) * 0.95

    # ── EQ correction (Modified IR) ───────────────────────────────────────────
    nn3 = np.arange(10 * fs + 1, 20 * fs)
    MS  = mic[nn3]
    PS  = np.convolve(pic[nn3], IR2, 'same')
    p,  cf, _, _, f1, f2 = cuki_ir_cli.oct_spectrum2(MS / np.amax(np.absolute(MS)), fs)
    p2, _,  _, _, _,  _  = cuki_ir_cli.oct_spectrum2(PS / np.amax(np.absolute(PS)), fs)
    g0  = p - p2

    IRX = np.zeros(NbF); IRX[0] = 1
    IR1 = IR2.copy()
    for i in range(len(f1)):
        g      = 10 ** (g0[i] / 20)
        B, A   = signal.butter(2, [f1[i], f2[i]], btype='bandpass', fs=fs)
        IRX   += signal.lfilter(B, A, IRX) * (g - 1)
        IR1   += signal.lfilter(B, A, IR1) * (g - 1)
    IR1 = IR1 / np.amax(np.absolute(IR1)) * 0.95

    IR3 = (np.array([1] + [0] * (NbF - 1), dtype=float) + IR2) / 2

    # ── save WAVs ─────────────────────────────────────────────────────────────
    fmt    = str(fs / 1000)
    prefix = f"IR_{stem}_{fmt[0:2]}k_{NbF}"
    for ir, suffix in [(IR1, "_M"), (IR2, "_Std"), (IR3, "_Std_Bld")]:
        out = output_dir / f"{prefix}{suffix}.wav"
        print(f"Saving {out} …")
        sf.write(str(out), ir, fs, 'PCM_24')

    # ── save plots ────────────────────────────────────────────────────────────
    print("Generating plots …")
    spec_dir = output_dir / "spectrum_graphs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    cuki_ir_cli.save_ir_plot(IR2, fs, NbF, "IR Std",    "Std Spectrum",    spec_dir / f"{prefix}_Std.png")
    cuki_ir_cli.save_ir_plot(IR1, fs, NbF, "IR M-file", "M-file Spectrum", spec_dir / f"{prefix}_M.png")
    print(f"\n✅  Done! Output saved to: {output_dir.resolve()}")
    print("  _Std      : Standard algorithm")
    print("  _Std_Bld  : Standard with 50% raw pickup / 50% IR blend")
    print("  _M        : Modified process (usually clearer than the standard)")
    import subprocess
    subprocess.Popen(["open", str(output_dir.resolve())])


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import subprocess
    if sys.platform == "darwin":
        subprocess.Popen(
            ["osascript", "-e", 'tell application "Python" to activate'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
