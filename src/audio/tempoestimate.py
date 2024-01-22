# -*- coding: utf-8 -*-
"""TempoEstimate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FXbvZerg3wJVArd3f50RZdA1fhxX2Ed1
"""

pip install librosa

import librosa

audio_file = librosa.load('test.wav')

# load the audio file as a waveform in y
# store the sample rate as sr
y, sr = audio_file

# load the tempo tracker to get the estimated tempo of the track
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# convert the beat frames into timings
# beat_times will be an array of timestamps (in seconds) corresponding to detected beat events
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
beat_times

# visualize the musical note - 12 pitch classes
import librosa.display
import matplotlib.pyplot as plt

# Compute the chromagram from a waveform or power spectrogram.
chroma = librosa.feature.chroma_stft(y=y, sr=sr)


fig, ax = plt.subplots()

# Display the chromagram
# the energy in each chromatic pitch class as a function of time
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)

ax.set(title='Chromagram')
fig.colorbar(img, ax=ax)
plt.show()

# understand it better
import numpy as np

ccov = np.cov(chroma)
fig, ax = plt.subplots()
img = librosa.display.specshow(ccov, y_axis = 'chroma', x_axis = 'chroma', ax=ax)
ax.set(title='Chroma covariance')
fig.colorbar(img,ax=ax)