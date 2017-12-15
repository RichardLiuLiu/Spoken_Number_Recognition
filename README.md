# Spoken Number Recognition

## Objective
To recognize a single spoken number from zero to nine. To be specific, as one speaks a number (0 - 9), the program will recognize the correct number.

## Dataset
Thanks to the dataset from Pannous, http://pannous.net/files/spoken_numbers_pcm.tar.

The dataset includes 2850 `.wav` files of 15 different people (male and female) speaking number 0 - 9.

## Python libraries required
`tensorflow-gpu`, `keras`, `librosa`, `numpy`, ` matplotlib`, `pyaudio`, `h5py`

## Main Idea
Draw the spectrogram of each `.wav` file, and save as an image. In this way, the speech recognition problem is transfered into a image recognition problem.
