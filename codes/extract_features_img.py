'''
Created on 2017.12.9

@author: Richard
'''

import os
import librosa.display
import librosa.feature
import numpy as np
import matplotlib.pyplot as plt

number = '9'

def extract_mfcc(in_path, file, fmax, nMel):
    y, sr = librosa.load(in_path + file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    
    plt.figure(figsize=(3, 3), dpi=100)
    librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max), fmax=fmax)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('myRecording/mfcc_images_ts/' + number + '/' + file[:-3] + 'png', bbox_inches='tight', pad_inches=-0.1)
    
    plt.close()   
    return


count = 0       # number of files processed

in_path = 'myRecording/audio_ts/' + number + '/'       # input directory
# in_path = '222/'
for wavfile in os.listdir(in_path):
    
    # Input file
    S = extract_mfcc(in_path, wavfile, 8000, 256)
    
    # Count processed files
    count += 1
    if count % 20 == 0:
        print ('%d files processed.' % count)

print ('Done!\t%d files processed.' % count)
