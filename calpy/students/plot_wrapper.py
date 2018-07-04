from .. import dsp
from .. import plots
from ..utilities import read_wavfile
from ..entropy import symbolise_speech, entropy_profile
import numpy as np

def plot_prosody( file_name, features=["waveform", "mfcc", "pitch", "intensity", "dB"], num_plots=200, num_chunks=10, scaling=4, print_status=False ):
    '''Plots a multirow plot of various prosody features.

        Args:
            filename (str): path to the audio file.
            features (list(string)): list of features to be plotted.  Ignores nonexistent features.  Defaults to ["waveform", "mfcc", "pitch", "intensity", "pitch_hist", "dB"]
            num_plots (int): divide the intial wavform into num_plots pieces and create one plot each.  Defaults to 200.
            num_chunks (int): number of subdivisions of one plot.  Defaults to 10.
            scaling (int): scales the size of the output plot.

        Returns:
            null:  saves plots to  a folder in current directory.
    '''
    plots.all_profile_plot( file_name=file_name, features=features, num_plots=num_plots, num_chunks=num_chunks, scaling=scaling, print_status=print_status )

def plot_sounding_pattern(
        file_name_A,
        file_name_B,
        time_step=0.01,
        time_range=(0,-1),
        row_width=10,
        row_height=1,
        duration_per_row=60,
        xtickevery=10,
        ylabels='short',
        dpi=300,
        filename="sounding_pattern_plot",
        title="sounding_pattern" ):
    """Plot sounding patterns like uptakes, inner pauses, over takes.
    Args:
        file_name_A (str): path to audio file of speaker A.
        file_name_B (str): path to audio file of speaker B.
        time_step (float, optional): time interval in between two elements in seconds, default to 0.01s.
        time_range ((float, float), optional): time range of the plot in seconds, default to from the entire converstaion.
        row_width (int, optional): parametre for display purpose, the width of a row, default to 10 units.
        row_height (int, optional): parametre for display purpose, the height of a row, default to 1 unit.
        duration_per_row (float, optional): parametre for display purpose, the duration of a row in seconds, default to 60 seconds.
        xtickevery (float, optional): parametre for display purpose, the duration of time in seconds in between two neighbour x ticks, default to 10 secnds.
        ylabels (string, optional): self explanatory.
        dpi (int, optional): self explanatory.
        filename (string, optional): file name of the output figure.
        title (string, optional): self explanatory.

        
    Returns:
        True, and write figure to disk.
    """
    fs_A, sound_A = read_wavfile(file_name_A)
    fs_B, sound_B = read_wavfile(file_name_B)
    pause_A = dsp.pause_profile(sound_A, fs_A)
    pause_B = dsp.pause_profile(sound_B, fs_B)
    plots.sounding_pattern_plot(
        A=pause_A,
        B=pause_B,  
        time_step=time_step,
        time_range=time_range,
        row_width=row_width,
        row_height=row_height,
        duration_per_row=duration_per_row,
        xtickevery=xtickevery,
        ylabels=ylabels,
        dpi=dpi,
        filename=filename,
        title=title)

def plot_anomaly(file_name):
    fs, sound = read_wavfile(file_name)
    pause = dsp.pause_profile(sound, fs)
    pitch = dsp.pitch_profile(sound, fs)
    prosody = np.append([pitch], [pause], axis=0)
    symbols = np.array([])
    for arr in np.array_split(prosody, pause.shape[0] // 10, axis=1):
        symbols = np.append(symbols, symbolise_speech(arr[0,:], arr[1,:]))
    
    entropy_prof = entropy_profile(symbols, window_size=200,window_overlap=100)
    plots.profile_plot(entropy_prof, file_name="anomaly.png",figsize=(25,4))
