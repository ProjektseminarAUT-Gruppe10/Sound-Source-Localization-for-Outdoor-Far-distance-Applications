# =============================================================================
# Demo Nr. 1:
# Objekte wie Text, Kreis, Punkt, Linie zeichnen
# =============================================================================

import matplotlib.pyplot as plt

# Drawing Methods
def initDrawing(figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    return fig, ax

def finishDrawing(xmin, ymin, xmax, ymax, title, xlabel, ylabel):
    plt.gca().grid(zorder=0)
    plt.gca().set_xlim([xmin, xmax])
    plt.gca().set_ylim([ymin, ymax])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
 
def finishDrawingL(xmin, ymin, xmax, ymax, title, xlabel, ylabel):
    plt.gca().grid(zorder=0)
    plt.gca().set_xlim([xmin, xmax])
    plt.gca().set_ylim([ymin, ymax])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.show()
    
def drawCircle(point, radius, color):
    circle = plt.Circle(point, radius, fill=False, edgecolor=color)
    plt.gca().add_patch(circle)

def drawPoint(point, marker, color, size):
    # for marker styles see list https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
    plt.scatter(point[0], point[1], s=size, c=color, marker=marker, zorder=3)

def drawPointL(point, marker, color, size, label):
    # for marker styles see list https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
    plt.scatter(point[0], point[1], s=size, c=color, marker=marker, zorder=3, label=label)
              
def drawLine(pointA, pointB, style, color, size):
    # for line style see list https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    plt.plot([pointA[0], pointB[0]], [pointA[1], pointB[1]], color=color, linestyle=style, linewidth=size)    
    
def drawString(text, point, color, size):
    plt.gca().text(point[0], point[1], text, color=color, fontsize=size)

def drawCurve(X,Y, color, style, size):
    # for line style see list https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    plt.plot(X,Y, color=color, linestyle=style, linewidth=size)
    
def drawMarkerCurve(X,Y, color, Lstyle, Mstyle, Lsize, Msize):
    # for marker styles see list https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
    # for line style see list https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
    plt.plot(X,Y, color=color, linestyle=Lstyle, maker = Mstyle, linewidth=Lsize, markersize=Msize)

# =============================================================================
# # DEMO: a simple drawing demo
# =============================================================================
#import numpy as np
#
## Define Points
#pointA = np.asarray([1,0])
#pointB = np.asarray([0,1])
#
## Init Graphics
#fig, ax = initDrawing()
#
## Draw
#drawCircle(pointA, 2.0, "green")
#drawLine(pointA,pointB,"-","gray",1)
#drawPoint(pointA, "x", "blue", 50)
#drawPoint(pointB, "x", "blue", 50)
#drawString("TestA", pointA, "black",15)
#
## Finish Graphics
#finishDrawing(fig, ax, -4, -4, 4, 4, "Drawing Demo", "X-Achsenbeschriftung", "Y-Achsenbeschriftung")
