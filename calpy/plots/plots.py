import numpy
import bokeh.plotting
import bokeh.io
from ..utilities import utilities

def recurrence( AA, ID=numpy.empty(0,dtype=int), colours=["red","blue","green"] ):
    """
        Plots a recurrence plot.
        
        AA :: reccurence matrix
        ID :: vector so that speaker( col[i] ) = ID[i] 

        return :: bokeh plot object
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
    bokeh.plotting.show( bokeh_plot )
    return

def export( bokeh_plot, file_path, astype="png"):

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