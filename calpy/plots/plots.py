import numpy
import bokeh.plotting
import bokeh.io
from .. import utilities

def recurrence( AA, ID=numpy.empty(0,dtype=int), colours=["red","blue","green"] ):
    """Plots a recurrence plot.
        
        Args:
            AA (numpy.array(float)):  A  2D reccurence matrix.
            ID (numpy.array(int)):  A vector so that speaker( col[i] ) = ID[i].  Defaults to the 0 vector.
            colours (list(str)):  Colours for the plot.

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

<<<<<<< HEAD
    Args:
        bokeh_plot: (bokeh plot object)

    Returns:
        null:  Outputs a plot on the default plot device.
=======
        Args:
            bokeh_plot (bokeh plot object)

        Return
            null:  Outputs a plot on the default plot device.
>>>>>>> ae1d5ff3958ed40bf60d8e8ab0130a156ac63667
    """
    bokeh.plotting.show( bokeh_plot )
    return

def export( bokeh_plot, file_path, astype="png"):
    """Save a plot as picture file.

        Args:
            bokeh_plot (bokeh plot object): The plot object to be saved.
            file_path (str):  Where to save the picture.
            astype (str):  The file type.  Defaults to png.

<<<<<<< HEAD
    Returns:
        null:  Outputs a plot to a file.
=======
        Return
            null:  Outputs a plot to a file.
>>>>>>> ae1d5ff3958ed40bf60d8e8ab0130a156ac63667
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