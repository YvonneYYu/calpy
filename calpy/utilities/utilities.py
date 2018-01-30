#SHARED UTILITY TOOLS AND VARIABLES
import numpy
import scipy.io.wavfile

def pad_signal(signal, sampling_rate, time_step = 0.01, frame_window = 0.025):
    """
        segement a signal, 1D audio signal, into frames, such that:
        output: N by M matrix, in which:
            each row is a segment of frame_window's audio signal
    """

    T  = int(sampling_rate * time_step)
    Fr = int(sampling_rate * frame_window)
    Fr +=int(sampling_rate * frame_window > Fr)
    signal = numpy.append(signal, numpy.zeros(Fr-len(signal)%T))
    
    return signal

def compress_pause_to_time(signal, sampling_rate, time_step = 0.01, frame_window = 0.025):
    """
        compress pause index to time
        input:
            signal: 1D boolean numpy array (padded already), True indicating pause
            sampling_rate: in Hz :: a float
            time_step: temporal step in seconds:: a float
            frame_window frame size is 25 ms by default
        output:
            pause: aligned pause :: 1D boolean numpy array
    """
    
    T  = int(sampling_rate * time_step)
    Fr = int(sampling_rate * frame_window)
    Fr +=int(sampling_rate * frame_window > Fr)
    
    length = (len(signal) - Fr)//T + 1
    pause  = numpy.full( length, False )

    for i in range(length):
        if len(numpy.where(signal[i*T:(i+1)*T])[0]) > T/2:
            pause[i] = True
    
    return pause

def is_upper_triangular( AA ):
    return numpy.allclose(AA, numpy.triu(AA))

def is_lower_triangular( AA ):
    return numpy.allclose(AA, numpy.tril(AA))

def read_wavfile( filename, channel=0 ):
    sampling_rate, datas = scipy.io.wavfile.read(filename)
    datas = datas.astype(float)
    
    if channel > len( datas.shape ):
        print("Error: Channel {} does not exist.  Note: first channel is channel 0.".format(channel))
        return
    elif len(datas.shape)>1:
        return sampling_rate, datas[:,channel]
    
    return sampling_rate, datas

def merge_pitch_profile( pitches, speaker_id ):
    """
        merges n-pitch profiles into one sound based on speaker_id
    """

    N = len( speaker_id )

    merged_pitch_profile = numpy.empty( N )
    for i in range(N):
        merged_pitch_profile[i] = pitches[speaker_id[i]][i]

    return merged_pitch_profile