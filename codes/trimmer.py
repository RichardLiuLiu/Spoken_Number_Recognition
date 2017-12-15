'''
Created on 2017.11.26

@author: Richard
'''

import wave
import struct
import os

DURATION = 1    # extend wave files to 1 second
count = 0       # number of files processed

read_path = 'recordings/'   # input directory
write_path = 'train/'       # output directory

for input_wavfile in os.listdir(read_path):
    
    # Input file
    input_wf = wave.open(read_path + input_wavfile, 'rb')
    RATE     = input_wf.getframerate()
    WIDTH    = input_wf.getsampwidth()
    CHANNELS = input_wf.getnchannels()
    
    # Output file
    output_wf = wave.open(write_path + input_wavfile, 'w')
    output_wf.setframerate(RATE)
    output_wf.setsampwidth(WIDTH)
    output_wf.setnchannels(CHANNELS)
    
    LEN = RATE * DURATION
    # Get first frame
    input_string = input_wf.readframes(1)
    for i in range(0, LEN):
        
        # Fill the output with 0 if there is no input
        if len(input_string) == 0:
            output_value = 0
            output_string = struct.pack('h', output_value)
            output_wf.writeframes(output_string)
            input_string = input_wf.readframes(1)
            continue
        
        # Get the input and output
        input_tuple = struct.unpack('h', input_string)
        output_value = input_tuple[0]

        # Save the wave file
        output_string = struct.pack('h', output_value)
        output_wf.writeframes(output_string)

        # Get next frame
        input_string = input_wf.readframes(1)
    
    input_wf.close()
    output_wf.close()
    
    # Count processed files
    count += 1
    if count % 50 == 0:
        print ('%d files processed.' % count)

print ('%d files processed.' % count)