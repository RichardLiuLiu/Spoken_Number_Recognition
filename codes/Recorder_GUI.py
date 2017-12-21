'''
Created on 2017.12.20

@author: Richard
'''

import numpy as np
import pyaudio
import struct
import wave
import sys
import tkinter as Tk
top = Tk.Tk()

import keras.backend as K
K.clear_session()

import librosa.display
import librosa.feature
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

nrow = 200
ncol = 200

BLOCKSIZE = 128

RATE = 22050
WIDTH = 2
CHANNELS = 1
LEN = 1 * RATE

model = models.load_model('mfcc_cnn_model_all.h5')

def is_silent(data, THRESHOLD):
    "Returns 'True' if below the threshold"
    return max(data) < THRESHOLD

def extract_mfcc(file, fmax, nMel):
    y, sr = librosa.load(file)
    
    plt.figure(figsize=(3, 3), dpi=100)
#    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
#    librosa.display.specshow(D)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max), fmax=fmax)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('tmp/tmp/myImg.jpg', bbox_inches='tight', pad_inches=-0.1)
    plt.close()
    
    return

def predict():
    # MFCCs of the test audio
    extract_mfcc('myNumber.wav', 8000, 256)
    test_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False)
    test_generator = test_datagen.flow_from_directory(
            'tmp',
            target_size=(nrow, ncol),
            batch_size=1,
            class_mode='sparse')

    # Load the model
    Xts, _ = test_generator.next()

    # Predict the probability of each class
    yts = model.predict(Xts)
#    print (yts * 100)
    if np.max(yts) < 0.1:
        print ('Cannot Recognize!')
        s1.set('Cannot Recognize!')
        top.update()
        return

    # Choose the most likely class
    res = np.argmax(yts)
    print (res)
    s1.set('You said: '+ str(res))
    top.update()
    return


##########  Recorder  ###########
def record():
    for i in range(10):
        # Output wave file
        output_wf = wave.open('myNumber.wav', 'w')
        output_wf.setframerate(RATE)
        output_wf.setsampwidth(WIDTH)
        output_wf.setnchannels(CHANNELS)

        p = pyaudio.PyAudio()
        stream = p.open(format = p.get_format_from_width(WIDTH),
                        channels = CHANNELS,
                        rate = RATE,
                        input = True,
                        output = True)
    
        # Wait until voice detected
        while True:    
            input_string = stream.read(BLOCKSIZE, exception_on_overflow = False)
            input_value = struct.unpack('h' * BLOCKSIZE, input_string)
            silent = is_silent(input_value, 500)
            if not silent:
                break

        # Start recording
        print("Start")
        
        nBLOCK = int(LEN / BLOCKSIZE)
        numSilence = 0
        
        for n in range(0, nBLOCK + 1):
        
            if is_silent(input_value, 100):
                numSilence += 1
 
            output_value = np.array(input_value)
        
            if numSilence > 20:
                break
        
            output_value = output_value.astype(int)
            output_value = np.clip(output_value, -2**15, 2**15 - 1)

            ouput_string = struct.pack('h' * BLOCKSIZE, *output_value)
            output_wf.writeframes(ouput_string)
        
            input_string = stream.read(BLOCKSIZE, exception_on_overflow = False)
            input_value = struct.unpack('h' * BLOCKSIZE, input_string)

        print('Done')
    
        predict()
    
        stream.stop_stream()
        stream.close()
        p.terminate()
        output_wf.close()


s1 = Tk.StringVar()
s1.set('You said: ')
L0 = Tk.Label(top, text = 'Spoken Number Recognition', font = (None, 30))
B1 = Tk.Button(top, text = 'Start', command = record, font = (None, 20))
L1 = Tk.Label(top, textvariable = s1, font = (None, 25))
B3 = Tk.Button(top, text = 'Quit', command = quit, font = (None, 20))  

L0.pack()
B1.pack()
B3.pack()
L1.pack(fill = Tk.X)

top.mainloop()

