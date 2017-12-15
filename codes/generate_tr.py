import numpy as np
import pyaudio
import struct
import wave

BLOCKSIZE = 128

# Get recording parameters
wf0 = wave.open('train/0_jackson_0.wav', 'rb')
RATE     = wf0.getframerate()
WIDTH    = wf0.getsampwidth()
CHANNELS = wf0.getnchannels()
LEN      = wf0.getnframes()
wf0.close()

def is_silent(data, THRESHOLD):
    "Returns 'True' if below the threshold"
    return max(data) < THRESHOLD

def record(path):
    
    # Output wave file
    output_wf = wave.open('myRecording/' + path, 'w')
    # output_wf = wave.open('222/-1_richard_22.wav', 'w')
    output_wf.setframerate(48000)
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
        silent = is_silent(input_value, 1000)
        if not silent:
            break

        # Start recording
    print("Start")

    nBLOCK = int(LEN / BLOCKSIZE)
    numSilence = 0
    for n in range(0, nBLOCK + 1):
        
        if is_silent(input_value, 100):
            numSilence += 1
#            output_value = np.zeros(BLOCKSIZE) 
 
        output_value = np.array(input_value)
        
        if numSilence > 5:
            break
        
        output_value = output_value.astype(int)
        output_value = np.clip(output_value, -2**15, 2**15 - 1)

        ouput_string = struct.pack('h' * BLOCKSIZE, *output_value)
        output_wf.writeframes(ouput_string)
        
        input_string = stream.read(BLOCKSIZE, exception_on_overflow = False)
        input_value = struct.unpack('h' * BLOCKSIZE, input_string)

    print('Done')

    stream.stop_stream()
    stream.close()
    p.terminate()
    output_wf.close()

record('audio_ts/9/9_yicong_30.wav')