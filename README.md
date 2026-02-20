# Cuki-IR-generator-Python

Python version of Cuki IR generator Light

The program is aimed at helping you to get a good plugged in tone out of your acoustic-electric guitar. Any acoustic guitar with an undersaddle transducer (UST) or sounboard transducer (SBT)
from companies like Fishman, LR Baggs, K&K or Martin, Taylor (with ES2), Gibson, Yamaha, Guild, Fedner, Takamine, Cort, Ibanez... should work.
It requires an IR loader, like a TC electronic Impulse IR loader, BOSS IR-200, Line 6 HX pedal or Torpedo CAB M (see my web site: acousticir.free.fr)

This program generates an IR wav file to transform your piezo pickup tone into an external microphone tone.

The input is a 1-2 min stereo file with

- Microphone take on the right
- Pickup take on the left
  The two tracks must have been recorded simultaneously with a 2-inputs audio interface or alike.

Note that the frequency sampling rate (44.1 KHz, 48 KHz or 96 KHz) is left unchanged. So if your pedal needs a specific format, pleasre record the stereo file in that format.

## Installation of dependencies

`uv` is recommended to install the python dependencies and run the program. https://docs.astral.sh/uv/guides/install-python/

- Clone this repository
- run `uv sync`

This should install all dependencies and allow you to run the program.

## Running cuki-ir-generator (CLI)

CLI: `uv run cuki-ir-generator <input.wav>`

You should see something like this, and the output files should be in the `output` directory:

```
➜  Cuki-IR-generator-Python git:(main) ✗ uv run cuki-ir-generator ~/Desktop/irtestinput.wav
Loading /Users/vic/Desktop/irtestinput.wav …
Computing IR (NbF=2048) …
Saving output/IR_irtestinput_44k_2048_M.wav …
Saving output/IR_irtestinput_44k_2048_Std.wav …
Saving output/IR_irtestinput_44k_2048_Std_Bld.wav …
Generating plots …
  graph saved → output/spectrum_graphs/IR_irtestinput_44k_2048_Std.png
  graph saved → output/spectrum_graphs/IR_irtestinput_44k_2048_M.png

✅ Done!
  _Std      : Standard algorithm
  _Std_Bld  : Standard with 50% raw pickup / 50% IR blend
  _M        : Modified process (usually clearer than the standard)
```

See `uv run cuki-ir-generator --help` for more options.

## Running cuki-ir-generator-gui (GUI version)

Some may find a GUI easier to use. To launch the GUI version: `uv run cuki-ir-generator-gui` (a tkinter GUI window will open that will guide you through the IR generation process)

<a href="https://ibb.co/LDGKBxKj"><img src="https://i.ibb.co/V0K6bg6n/Screenshot-2026-02-19-at-11-54-40-PM.png" alt="Screenshot-2026-02-19-at-11-54-40-PM" border="0"></a>

## Installation of cuki-ir-generator

If you would like to install the program, you can use pip:

`pip install .`

This will put the program in your PATH and allow you to run it from anywhere with simply: `cuki-ir-generator <input.wav>` or `cuki-ir-generator-gui` for the GUI version.

## Output files

Output files name are for example:

IR_XXXX_44k_2048_Std.wav
IR_XXXX_44k_2048_Std_Bld.wav
IR_XXXX_44k_2048_M.wav
where "44k" stands for 44.1KHz sampling rate. It is the ame as your recording.

where "2048" stands for the IR length (Cheap pedal only use 1024 or 512 pts from the 2048pts)

where "Std" stands for standard process. The process is a "light version" of the one used to produce the IR database in acousticir.free.fr

where "Std_Bld" stands for Standard with 50% raw pickup / 50% IR blend

where "M" stands for Modified process (usually clearer and brighter than the standard)

## Other python formats

The program can be found in Google Colab notebook format
"Cuki_IR_light.ipynb"

or in python file format
"Cuki_IR_light3.py"
