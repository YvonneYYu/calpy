import numpy

import os
import datetime

import bokeh.plotting
import bokeh.io

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

from .. import utilities
from .. import dsp
from .support import *

def recurrence( AA, ID=numpy.empty(0,dtype=int), colours=["red","blue","green"] ):
    """Plots a recurrence plot.
        
        Args:
            AA (numpy.array(float)):  A  2D reccurence matrix.
            ID (numpy.array(int), optional):  A vector so that speaker( col[i] ) = ID[i].  Defaults to the 0 vector.
            colours (list(str), optional):  Colours for the plot.

        Returns:
            bokeh plot object
    """

    isLower = utilities.is_lower_triangular(AA)
    AA = AA/AA.max()

    COLS = colours

    N = AA.shape[0]
    if not ID.size:
        ID = numpy.zeros(N,dtype=int)

    xs, ys, cols, alphas = list(), list(), list(), list()
    
    cell_padding = .8

    # Note:  r and c are deliberately swapped so that the transpose of AA is plotted
    for c in range(N): 
        for r in range(c+1 if isLower else N):
        #for r in range(N):
            #triangle bottom
            xs.append([r,r+cell_padding,r+cell_padding])
            ys.append([c,c,c+cell_padding])
            cols.append( COLS[ID[r]] if AA[c][r]>=0 or c==r else "pink" )
            alphas.append( 1-AA[c,r] )

            #triangle top
            xs.append([r,r,r+cell_padding])
            ys.append([c,c+cell_padding,c+cell_padding])
            cols.append( COLS[ID[c]] if AA[c][r]>=0 or c==r else "black" )
            alphas.append( 1-AA[c,r] )


    #Plot tweaks
    plot = bokeh.plotting.figure(
        plot_width=900,
        plot_height=900,
        min_border=100,
        y_range=(N,0),
        x_range=(0,N)
        )

    plot.toolbar.logo = None
    plot.toolbar_location = None
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None

    plot.patches( xs, ys, color=cols, alpha=alphas )

    return plot

def show( bokeh_plot ):
    """Print a plot to the screen.

        Args:
            bokeh_plot (bokeh plot object): bokeh plot object.

        Return
            null:  Outputs a plot on the default plot device.
    """
    bokeh.plotting.show( bokeh_plot )
    return

def export( bokeh_plot, file_path, astype="png"):
    """Save a plot as picture file.

        Args:
            bokeh_plot (bokeh plot object): The plot object to be saved.
            file_path (str):  Where to save the picture.
            astype (str, optional):  The file type.  Defaults to png, another option is svg.

        Return
            null:  Outputs a plot to a file.
    """
    if astype not in ["svg","png"]:
        print("Export type not supported.  Use 'svg' or 'png' only.")
        return

    if astype=="png":
        bokeh.io.export_png( bokeh_plot, filename=file_path+"."+astype)
        return

    if astype=="svg":
        bokeh_plot.output_backend = "svg"
        bokeh.io.export_svgs( bokeh_plot, filename=file_path+"."+astype)
        return

def profile_plot( ys, xlabel="", ylabel="", file_name="", figsize=(8,4), remove_zeros=False ):
    """Plots points on the plane and connects with a line.
    
        Args:
            ys (numpy.array(floats)):  List of numeric values to be plotted.
            xlabel (str, optional):  The name for the x-axis.  Defaults to emtpy.
            ylabel (str, optional):  The name for the y-axis.  Defaults to emtpy.
            file_name (str, optional):  Outputs picture to this file_name.  Defaults to empty.
            figsize (tuple(float,float), optional):  A tuple specifying (width, height) in inches of plot.  Defaults to (8,4)
            remove_zeros (bool, optional):  Toggle for replacing 0s with NANs.  Defaults to False.

        Returns:
            null : Saves an image to file_name else displays to default plot
    """
    if remove_zeros:
        ys[numpy.where(ys == 0)] = numpy.nan

    #define plot size in inches (width, height) & resolution(DPI)
    fig = plt.figure( figsize=figsize )
    
    plt.plot( ys, 'm-o',  ms=3 )
        
    if ylabel:
        plt.ylabel(ylabel)

    if xlabel:
        plt.xlabel(xlabel)

    if file_name:
        plt.savefig(file_name,dpi=300)
    else:
        plt.show()
    
    return 

def mfcc_plot( AA, file_name="mfcc_plot.png", figsize=(16,4) ):
    """Plots points on the plane and connects with a line.
    
        Args:
            AA (numpy.array(floats)):  2D array containing the MFCC.
            file_name (str, optional):  Outputs picture to this file_name.  Defaults to "mfcc_plot.png".
            figsize (tuple(float,float), optional):  A tuple specifying (width, height) in inches of plot.  Defaults to (16,4)

        Returns:
            null : Saves an image to file_name else displays to default plot
    """

    plt.figure(figsize=figsize)
    plt.imshow( AA, cmap = 'hot', alpha = 1, aspect="auto")
    plt.tight_layout()
    plt.axis('off')

    plt.savefig( file_name )
    plt.close()
    return

def _heatmap_dist( xs, num_bins=7, num_chunks=10 ):
    """A helper function for all_profile_plot that returns a matrix that encodes a pitch distribution as a heatmap.

        Args:
            xs (numpy.array(float)) :  A numpy array of floats.
            num_bins : Number of bins for historgram.  Defaults to 7.
            num_chunks:  Number of chunks of all_profile_plot.  Defaults to 10.

        Returns:
            numpy.matrix : A matrix for heat mapping. 
    """
    AA = numpy.matrix([0 for k in range(num_bins)])
    for k, chunk in enumerate( numpy.array_split(xs,num_chunks) ):
        #hist, bin_edges = numpy.histogram( chunk, bins=num_bins )
        hist, _ = numpy.histogram( chunk, density=True, range=(1,256), bins=num_bins )
        AA = numpy.vstack( (AA, numpy.matrix(hist)) )
    AA = AA.T
    AA = numpy.delete( AA, 0, 1 )
    return AA

def _add_lines( ax, num_chunks ):
    """A helper function for all_profile_plot that adds lines to indicate plot chunks.

        Args:
            ax (matplotlib axis):  An axis.
            num_chunks (int):  Number of chunks to be illustrated.

        Returns:
            null:  Adds lines to input axis.
    """
    xmin, xmax = ax.get_xlim()
    for L in numpy.arange(xmin+(xmax-xmin)/num_chunks, xmax, (xmax-xmin)/num_chunks):
        ax.axvline(x=L, color='m', linewidth=2.0 )
    return

def all_profile_plot( file_name, features=["waveform", "mfcc", "pitch", "pitch_hist", "dB"], num_plots=200, num_chunks=10, scaling=4, print_status=False ):
    '''Plots a multirow plot of various features.

        Args:
            filename (str): path to the audio file.
            features (list(string)): list of features to be plotted.  Ignores nonexistent features.  Defaults to ["waveform", "mfcc", "pitch", "intensity", "pitch_hist", "dB"]
            num_plots (int): divide the intial wavform into num_plots pieces and create one plot each.  Defaults to 200.
            num_chunks (int): number of subdivisions of one plot.  Defaults to 10.
            scaling (int): scales the size of the output plot.

        Returns:
            null:  saves plots to  a folder in current directory.
    '''
    available_features = ["waveform", "mfcc", "pitch", "pitch_hist", "dB"]
    features = [ feat for feat in features if feat in available_features ]  #Remove features that do not exist.

    fs, sound = utilities.read_wavfile(file_name)
    file_name = os.path.splitext(os.path.basename(file_name))[0]  #filename without path and extension

    num_feat = len(features)
    N        = len(sound)
    stride   = N//num_plots

    if not os.path.exists(file_name): 
        os.mkdir(file_name)

    for k in range(0, num_plots):

        fig = plt.figure( figsize=(8*scaling, num_feat*scaling) )
        plot_num = 0

        L, R  = stride*k, stride*(k+1)
        sound_chunk = sound[L:R]

        xs = list( L/fs+n/fs for n in range(L,R) )

        #feature:  Waveform
        if "waveform" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.tick_top()
            ax.title.set_visible(False)
            ax.plot( xs, sound_chunk )
            ax.margins(0)
            _add_lines( ax, num_chunks=num_chunks )

        #feature:  dB profile
        if "dB" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.title.set_visible(False)
            IP = dsp.dB_profile( sound_chunk, fs )
            ax.plot( IP, color='b', linestyle='--', marker='o' )
            ax.margins(0)
            _add_lines( ax, num_chunks=num_chunks )

        #feature:  Pitch profile
        if "pitch" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.title.set_visible(False)
            ax.set_ylim(0,256)
            PP = dsp.pitch_profile( sound_chunk, fs )
            ax.plot( PP, color='r', linestyle='--', marker='o' )
            ax.margins(0)
            _add_lines( ax, num_chunks=num_chunks )

        #feature:  Pitch hist
        if "pitch_hist" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.title.set_visible(False)
            AA = _add_lines( PP, num_chunks=num_chunks )
            ax.imshow( AA, cmap=plt.cm.Blues, alpha = 1, aspect="auto")
            _add_lines( ax, num_chunks )

        #feature:  MFCC
        if "mfcc" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.title.set_visible(False)
            AA = dsp.mfcc_profile( sound_chunk, fs )
            plt.imshow( AA, cmap = 'hot', alpha = 1, aspect="auto")
            _add_lines( ax, num_chunks=num_chunks )

        plt.subplots_adjust(hspace=.05)

        #save plot
        plt.savefig('{}/time{}_{}.png'.format(file_name, L, R), 
            bbox_inches='tight', 
            orientation="landscape", 
            dpi=200)
        plt.close()

        if print_status:
            print("Plot {} of {} saved to {}".format(k+1, num_plots, '{}/{}/time{}_{}.png'.format(os.getcwd(),file_name, L, R)))

def feature_distribution(features, output_file, bins=100, showfig=False, savefig=True):
    """Plot histogram of features
    Args:
        features (numpy.array): 1D or 2D feature vector. If 2D, features are along axis 1.
        output_file (string): path to output figure.
        bins (int, optional): number of bins, defaults to 100.
        showfig (bool, optional): True indicates to plot out the figure.  Defaults to False.
        savefig (bool, optional): True indicates to write the figure to disk.  Defaults to True.
    
    Returns:
        None, save a figure to output_file or show a figure if showfig.
    """
    assert showfig or savefig, "showfig and savefig parametres are both False, please set at least one to be True."
    # number of features.
    n = features.shape[0]
    fig = plt.figure()
    for i in range(n):
        hist, bins = numpy.histogram(features[i,:], bins=bins, density=False)
        plt.plot(bins[:-1], hist, '+-', label='feature' + str(i))
    plt.title('Histogram of all features')
    plt.legend()
    if savefig:
        fig.savefig(output_file, dpi=600)
    if showfig:
        fig.show()

def sounding_pattern_plot(
        A,
        B,  
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
        A (1D numpy.array): pauses of speaker A, with 1 indicates sounding.
        B (1D numpy.array): pauses of speaker B, with 1 indicated sounding.
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

    colours = ['white', 'lawngreen', 'lightsalmon', 'lightblue', 'darkorchid','tomato', 'cornflowerblue']

    if time_range[1] < 0:
        time_range = (time_range[0], A.shape[0]*time_step+1)

    total_duration = int(time_range[1]-time_range[0])#s
    #print("Total duration = {}".format(total_duration))

    A = A[ int(time_range[0]/time_step) : int(time_range[1]/time_step) ]
    B = B[ int(time_range[0]/time_step) : int(time_range[1]/time_step) ]

    num_subplot_rows = round( total_duration/duration_per_row )

    As = numpy.array_split(A, num_subplot_rows)
    Bs = numpy.array_split(B, num_subplot_rows)

    plt.figure(figsize=(row_width, row_height*num_subplot_rows), dpi=dpi )

    for plot_row in range(num_subplot_rows):
        A, B = As[plot_row], Bs[plot_row]
        ncols = A.shape[0]

        #Build matrix for heat mapping
        AA = numpy.zeros(shape=(3,ncols))

        AA[0,:] = colours.index('cornflowerblue')*A
        AA[2,:] = colours.index('tomato')*B

        AA[1, numpy.where( A ) ] = colours.index('cornflowerblue')
        AA[1, numpy.where( B ) ] = colours.index('tomato')
        AA[1, numpy.where( A & B ) ] = colours.index('darkorchid')

        #AupB
        conditions = ranges_satisfying_condition( A, B, *name_to_edge_condition["AupB"] )
        for L,R in conditions:
            AA[0:2, L:R+1] = colours.index('lawngreen')

        #BupA
        conditions = ranges_satisfying_condition( A, B, *name_to_edge_condition["BupA"] )
        for L,R in conditions:
            AA[1:3, L:R+1] = colours.index('lawngreen')

        #AipB
        conditions = ranges_satisfying_condition( A, B, *name_to_edge_condition["AipB"] )
        for L,R in conditions:
            AA[0, L:R+1] = colours.index('lightblue')

        #BipA
        conditions = ranges_satisfying_condition( A, B, *name_to_edge_condition["BipA"] )
        for L,R in conditions:
            AA[2, L:R+1] = colours.index('lightsalmon')

        #Plot row
        if title and plot_row==0:
            #plt.suptitle breaks tight_layout so give the first row a title instead
            ax = plt.subplot(num_subplot_rows, 1, plot_row+1, title=title)
        else:
            ax = plt.subplot(num_subplot_rows, 1, plot_row+1)

        #Set y coordinate axis
        ax.set_yticks( [] )
        ax.set_yticklabels( [] )

        #Set x axis
        if xtickevery < float('inf'):  #turn off ticks
            x_range = range(0, ncols+1, int(xtickevery/time_step))
        else:
            x_range = []
        ax.set_xticks( x_range )
        #Convert seconds to HR:MIN:SEC format
        to_time_format = lambda x : str(datetime.timedelta(seconds=x*time_step+time_range[0]+plot_row*duration_per_row))
        ax.set_xticklabels([to_time_format(x) for x in x_range])

        plt.imshow(
            AA, 
            cmap=LinearSegmentedColormap.from_list('mycmap', [*colours], N=len(colours) ), 
            aspect='auto',
            interpolation="nearest"
            )
        
    #plt.margins(1)
    plt.tight_layout()
    plt.savefig('{}.png'.format(filename))
    
    return True