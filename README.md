# Cuki-IR-generator-Python
Python version of Cuki IR generator Light

The program is aimed at helping you to get a good plugged in tone out of your acoustic-electric guitar. Any acoustic guitar with an undersaddle transducer (UST) or sounboard transducer (SBT)
from companies like Fishman, LR Baggs, K&K or Martin, Taylor (with ES2), Gibson, Yamaha, Guild, Fedner, Takamine, Cort, Ibanez... should work.
It requires an IR loader, like a TC electronic Impulse IR loader, BOSS IR-200, Line 6 HX pedal or Torpedo CAB M (see my web site: acousticir.free.fr)

This program generates an IR wav file to transform your piezo pickup tone into an external microphone tone.

The input is a 1-2 min stereo file with 
* Microphone take on the left
* Pickup take on the right
The two tracks must have been recorded simultaneously with a 2-inputs audio interface or alike.

Note that the frequency sampling rate (44.1 KHz, 48 KHz or 96 KHz) is left unchanged. So if your pedal needs a specific format, pleasre record the stereo file in that format.

Change the filename in the 5th line of the code to match your file name

filename='XXXX.wav'

Output files name are for example:

IR_XXXX_44k_2048_Std.wav
IR_XXXX_44k_2048_Std_Bld.wav
IR_XXXX_44k_2048_M.wav
where "44k" stands for 44.1KHz sampling rate. It is the same as your recording.

where  "2048" stands for the IR length (Cheap pedal only use 1024 or 512 pts from the 2048pts)

where "Std" stands for standard process. The process is a "light version" of the one used to produce the IR database in acousticir.free.fr

where "Std_Bld" stands for Standard with 50% raw pickup / 50% IR blend

where "M" stands for Modified process (usually clearer and brighter than the standard)

The program can be found in Google Colab notebook format
"Cuki_IR_light.ipynb"

or in python file format
"Cuki_IR_light3.py"

Dependencies
Required librairies are:
soundfile
math
numpy
scipy

Optional librairy is
matplotlib
