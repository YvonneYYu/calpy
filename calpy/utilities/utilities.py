#SHARED UTILITY TOOLS AND VARIABLES
import numpy
import scipy.io.wavfile

def pad_signal(signal, sampling_rate, time_step = 0.01, frame_window = 0.025):
    """segement a signal, 1D audio signal, into frames, such that:
        output: N by M matrix, in which:
            each row is a segment of frame_window's audio signal
    """

    T  = int(sampling_rate * time_step)
    Fr = int(sampling_rate * frame_window)
    Fr +=int(sampling_rate * frame_window > Fr)
    signal = numpy.append(signal, numpy.zeros(Fr-len(signal)%T))
    
    return signal

def compress_pause_to_time(signal, sampling_rate, time_step = 0.01, frame_window = 0.025):
    """compress pause index to time
        Args:
            signal (numpy.array(bool)): A list of pause sequence. True indicating pause.
            sampling_rate (int): sampling frequency in Hz.
            time_step (float, optional): The time interval (in seconds) between two pauses. Default to 0.01.
            frame_window (float, optional): The length of speech (in seconds) used to estimate pause. Default to 0.025.
        Returns:
            numpy.array(bool): compressed pause.
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
    """Check if a matrix is upper triangular.
        Args:
            AA (numpy.array): a 2D matrix.
        Returns:
        bool:
    """
    return numpy.allclose(AA, numpy.triu(AA))

def is_lower_triangular( AA ):
    """Check if a matrix is lower triangular.
        Args:
            AA (numpy.array): a 2D matrix.
        Returns:
        bool:
    """
    return numpy.allclose(AA, numpy.tril(AA))

def read_wavfile( filename, channel=0 ):
    """Read in a audio file (in .wav format) and enforce the output as mono-channel.
        Args:
            filename (str): path to the audio file.
            channel(int, optional): indicate which channel to read in. Defaults to 0.
        Returns:
            int: sampling frequency.
            numpy.array: audio data.
    """
    sampling_rate, datas = scipy.io.wavfile.read(filename)
    datas = datas.astype(float)
    
    if channel > len( datas.shape ):
        print("Error: Channel {} does not exist.  Note: first channel is channel 0.".format(channel))
        return
    elif len(datas.shape)>1:
        return sampling_rate, datas[:,channel]
    
    return sampling_rate, datas

def write_wavfile(filename, fs, data):
    scipy.io.wavfile.write(filename, fs, data)

def merge_pitch_profile( pitches, speaker_id ):
    """merges n-pitch profiles into one sound based on speaker_id.

        Args:
            pitches (list-like(float)): a sequence of pitches.
            speaker_id (list-like(int)): a list of speakers' id.
        Returns:
            numpy.array: merged pitch profile.
    """

    N = len( speaker_id )

    merged_pitch_profile = numpy.empty( N )
    for i in range(N):
        merged_pitch_profile[i] = pitches[speaker_id[i]][i]

    return merged_pitch_profile

def artificial_signal( frequencys, sampling_frequency=16000, duration=0.025 ):
    """Concatonates a sequence of sinusoids of frequency f in frequencies.

        Args:
            frequencys (list-like(int)): sequence of frequencies of sinusoidual signals in Hz.
            sampling_frequency (int, optional): sampling frequency in Hz. Defaults to 16000.
            duration (float, optional): duration of the output sinusoid in seconds. Defaults to 0.025.
        Returns:
            numpy.array: artificially generated sinusoidal signal.
    """
    sins = map( lambda f : sinusoid(f, sampling_frequency, duration), frequencys)
    return numpy.concatenate( tuple(sins) )

def sinusoid( frequency, sampling_frequency=16000, duration=0.025 ):
    """Generate a sinusoid signal.
        Args:
            frequency (int): the frequency of the sinusoidal signal.
            sampling_frequency (int, optional): sampling frequency in Hz. Defaults to 16000.
            duration (float, optional): duration of the output sinusoid in seconds. Defaults to 0.025.

        Returns:
        numpy.array: a sinusoid.
    """
    times = numpy.arange(int(sampling_frequency * duration))
    return numpy.sin(2 * numpy.pi * frequency * times / sampling_frequency)

def random_symbols( distribution, length ):
    if sum(distribution) != 1:
        print("Warning: probabilites must sum to 1")
        return

    return numpy.random.choice( len(distribution), length, p=distribution )

def random_run( distributions, length, min_run=100, max_more=100 ):
    ans  = list()
    k, N, M = 0, length, len(distributions)
    
    while True:
        more = numpy.random.randint(0,max_more) if max_more else 0
        ext_length = min_run + more
        ext_length = min( ext_length, N )
        
        ans.extend( random_symbols( distributions[k % M], ext_length ) )
        k += 1 % M

        N -= ext_length
        if N <= 0: return ans