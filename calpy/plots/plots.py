import numpy

import os

import bokeh.plotting
import bokeh.io

import matplotlib
import matplotlib.pyplot as plt

from .. import utilities
from .. import dsp

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
        plt.savefig(file_name)
    else:
        plt.show()
    
    return 

def mfcc_plot( AA, file_name="", figsize=(16,4) ):
    """Plots points on the plane and connects with a line.
    
        Args:
            AA (numpy.array(floats)):  2D array containing the MFCC.
            file_name (str, optional):  Outputs picture to this file_name.  Defaults to empty.
            figsize (tuple(float,float), optional):  A tuple specifying (width, height) in inches of plot.  Defaults to (8,4)

        Returns:
            null : Saves an image to file_name else displays to default plot
    """

    plt.figure(figsize=figsize)
    plt.imshow( AA, cmap = 'hot', alpha = 1, aspect="auto")
    plt.tight_layout()
    plt.axis('off')

    if file_name:
        plt.savefig( file_name )
    else:
        plt.show()
    
    return

def heatmap_dist( xs, num_bins=7, num_chunks=10 ):
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

def add_lines( ax, num_chunks ):
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

def all_profile_plot( file_name, features=["waveform", "mfcc", "pitch", "intensity", "pitch_hist", "dB"], num_plots=200, num_chunks=10, scaling=4, print_status=False ):
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
    available_features = ["waveform", "mfcc", "pitch", "intensity", "pitch_hist", "dB"]
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
            add_lines( ax, num_chunks=num_chunks )

        #feature:  dB profile
        if "dB" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.title.set_visible(False)
            IP = dsp.dB_profile( sound_chunk, fs )
            ax.plot( IP, color='b', linestyle='--', marker='o' )
            ax.margins(0)
            add_lines( ax, num_chunks=num_chunks )

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
            add_lines( ax, num_chunks=num_chunks )

        #feature:  Pitch hist
        if "pitch_hist" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.title.set_visible(False)
            AA = heatmap_dist( PP, num_chunks=num_chunks )
            ax.imshow( AA, cmap=plt.cm.Blues, alpha = 1, aspect="auto")
            add_lines( ax, num_chunks )

        #feature:  MFCC
        if "mfcc" in features:
            plot_num += 1
            ax = plt.subplot(num_feat, 1, plot_num)
            ax.xaxis.set_visible(False)
            ax.title.set_visible(False)
            AA = dsp.mfcc_profile( sound_chunk, fs )
            plt.imshow( AA, cmap = 'hot', alpha = 1, aspect="auto")
            add_lines( ax, num_chunks=num_chunks )

        plt.subplots_adjust(hspace=.05)

        #save plot
        plt.savefig('{}/time{}_{}.png'.format(file_name, L, R), 
            bbox_inches='tight', 
            orientation="landscape", 
            dpi=200)
        plt.close()

        if print_status:
            print("Plot {} of {} saved to {}".format(k+1, num_plots, '{}/{}/time{}_{}.png'.format(os.getcwd(),file_name, L, R)))