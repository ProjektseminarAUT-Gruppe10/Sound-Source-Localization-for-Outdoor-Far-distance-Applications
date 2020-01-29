# Imports
import numpy as np
from GeometryLibrary import getPoint

# Arrays
def getMicrophonePositions_SIM_A(radius): # on circle
    points = list()
    points.append(getPoint(radius*np.cos(4*np.pi/5),radius*np.sin(4*np.pi/5)))
    points.append(getPoint(radius*np.cos(3*np.pi/5),radius*np.sin(3*np.pi/5)))
    points.append(getPoint(radius*np.cos(2*np.pi/5),radius*np.sin(2*np.pi/5)))
    points.append(getPoint(radius*np.cos(1*np.pi/5),radius*np.sin(1*np.pi/5)))
    return points

def getMicrophonePositions_SIM_B(radius): # in triangle
    points = list()
    points.append(getPoint(-radius,0))
    points.append(getPoint(0,+radius))
    points.append(getPoint(+radius,0))
    return points
    
def getMicrophonePositions_SIM_C(radius): # in straight line
    points = list()
    points.append(getPoint(-radius,0))
    points.append(getPoint(0,0))
    points.append(getPoint(+radius,0))
    return points

def getMicrophonePositions4_TDOA(points):
    if(len(points)==3):
        l = list()
        l.append(points[0])
        l.append(points[1])
        l.append(points[1])
        l.append(points[2])
        return l
    else:
        return points
    
def getMicrophonePositions4_AMP(points):
    l = list()
    l.append(points[0])
    l.append(points[len(points)-1])
    return l

def getMicrophoneSignals4_TDOA(signals):
    if(len(signals)==3):
        l = list()
        l.append(signals[0])
        l.append(signals[1])
        l.append(signals[1])
        l.append(signals[2])
        return l
    else:
        return signals
    
def getMicrophoneSignals4_AMP(signals):
    l = list()
    l.append(signals[0])
    l.append(signals[len(signals)-1])
    return l