# Spoken Number Recognition
Read `Project_Report.pdf` for more details.

## Objective
To recognize a single spoken number from zero to nine. To be specific, as one speaks a number (0 - 9), the program will recognize the correct number.

## Dataset
Thanks to the dataset from Pannous, http://pannous.net/files/spoken_numbers_pcm.tar.

The dataset includes 2850 `.wav` files of 15 different people (male and female) speaking number 0 - 9. Besides, 400 `.wav` files recorded by me and my roommate are added to the dataset.

## Python Libraries Required
`tensorflow-gpu`, `keras`, `librosa`, `numpy`, ` matplotlib`, `pyaudio`, `h5py`

## Main Idea
Draw the spectrogram of each `.wav` file, and save as an image. In this way, the speech recognition problem is transfered into an image recognition problem.

Use CNN to build a classifier for the dataset. The CNN model includes 2 Dense (fully connected) layers and 5 Convolution layers, with Max-Pooling and BatchNormalization layers in it.

## Evaluation
At the training stage, the acc reaches 100% after 20 epochs, and val_acc reaches 98%. Very nice!

At the real-time test stage, the error rate keeps very low. Nice!

## References
[1] https://github.com/libphy/which_animal

[2] https://github.com/pannous/tensorflow-speech-recognition

[3] https://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/
