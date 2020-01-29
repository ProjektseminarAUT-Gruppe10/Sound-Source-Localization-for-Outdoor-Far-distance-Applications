# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

sys.path.append("..\\..\\libraries")
from GraphicLibrary import drawPoint, drawCircle, initDrawing, finishDrawing
from GeometryLibrary import getPoint, angle_radians

filename = "out-dist-0,40m_B-complete.txt"

# Lade Daten
data = list()    
with open(filename) as csvDataFile:
    csvReader = csv.reader(csvDataFile, delimiter=';')    
    for row in csvReader:
        data.append(row)
data = data[1:]

# Stelle Ergebnisse da
initDrawing(figsize=(16,8))

import matplotlib.cm as cm
from matplotlib.colors import Normalize
cmap = cm.Reds #cm.autumn
norm = Normalize(vmin=0, vmax=1)

error1 = list()
error2 = list()
for d in data:
    error1.append(float(d[7]))
    error2.append(float(d[11]))
errmax1 = np.max(error1)
errmax2 = np.max(error2)

plt.subplot(1,2,1)
plt.grid()
for d in data:
    dist = float(d[0])
    angl = float(d[1])
    distError = float(d[7])
    relError = distError/dist
    point = getPoint(dist*np.sin(angle_radians(angl)),dist*np.cos(angle_radians(angl)))
    plt.scatter(point[0], point[1], s=1000*distError/errmax1, c="Black")
#plt.colorbar()

plt.subplot(1,2,2)
plt.grid()
for d in data:
    dist = float(d[0])
    angl = float(d[1])
    anglError = float(d[11])
    point = getPoint(dist*np.sin(angle_radians(angl)),dist*np.cos(angle_radians(angl)))
    plt.scatter(point[0], point[1], s=1000*anglError/errmax2, c="Black")
#plt.colorbar()
