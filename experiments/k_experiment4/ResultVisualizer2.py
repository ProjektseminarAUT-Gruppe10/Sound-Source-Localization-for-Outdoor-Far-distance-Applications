# Import
import sys
import math
import numpy as np

sys.path.append("..\\..\\simulation")
sys.path.append("..\\..\\libraries")
from GeometryLibrary import calculateMicrophoneArray_2
from SimulationLibrary import load_configs, simulate
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GeometryLibrary import getAngle_angle1, get_tCurve
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric
from GraphicLibrary import drawPoint, drawCircle, drawCurve, initDrawing, finishDrawing
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree

sys.path.append("..\\..\\tdoa")
import array_parameters
import basic_tdoa


def plausibleFilter_TDOA(solutions):
#    print(solutions)
    result = list()
    for p in solutions:
        if(p[1]>=0):
            result.append(p)
    if(len(result)==1):
        return result[0]
    elif(len(result)==2):
        if(np.linalg.norm(result[0])>np.linalg.norm(result[1])):
            return result[0]
        else:
            return result[1]    
    else:
        return "no"
    
def SSL_TDOA_LIN(tdoa1, tdoa2, micA, micB, micC, micD):        
    std1 = tdoa1*343.2
    std2 = tdoa2*343.2
    steep1 = KarstenDOA_calculateSteep_linear_simple(distance(micA, micB), std1)
    steep2 = KarstenDOA_calculateSteep_linear_simple(distance(micC, micD), std2)
    solutions, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micA, micB, micC, micD, steep1, steep2)
    estimation = plausibleFilter_TDOA(solutions)
    return estimation






array = "D"
mic_dist = 0.4

# Array errechnen
if(array=="A"):
    micA = getPoint(-mic_dist*3/2,0)
    micB = getPoint(-mic_dist*1/2,0)
    micC = getPoint(+mic_dist*1/2,0)
    micD = getPoint(+mic_dist*3/2,0)
if(array=="B"):
    micA = getPoint(-mic_dist*1/2,+mic_dist*1/2)
    micB = getPoint(+mic_dist*1/2,+mic_dist*1/2)
    micC = getPoint(-mic_dist*1/2,-mic_dist*1/2)
    micD = getPoint(+mic_dist*1/2,-mic_dist*1/2)
if(array=="C"):
    micA = getPoint(-mic_dist*3/2,0)
    micB = getPoint(-mic_dist*1/2,0)
    micC = getPoint(+mic_dist*1/2,0)
    micD = getPoint(+mic_dist*3/2,0)
if(array=="D"):
    micA = getPoint(-mic_dist*1/2,+mic_dist*1/2)
    micB = getPoint(-mic_dist*1/2,-mic_dist*1/2)
    micC = getPoint(+mic_dist*1/2,0)
    micD = getPoint(+mic_dist*3/2,0)
    
# Darstellen    
initDrawing(figsize=(16,8))
        
drawPoint(micA, "x", "blue", 50)
drawPoint(micB, "x", "blue", 50)
drawPoint(micC, "x", "blue", 50)
drawPoint(micD, "x", "blue", 50)

for tdoa1 in range(-100,100):
    for tdoa2 in range(-10,10):
        p = SSL_TDOA_LIN(tdoa1*1/46000, tdoa2*1/46000, micA, micB, micC, micD)
        if(p!="no"):
            drawPoint(p, ".", "black", 10)
#for tdoa in range(-30,30):
#    X,Y = KarstenDOA_calculateCurve_linear(micA, micB, 3*tdoa*1/48000*343.2, res=0.1, rang=100)
#    drawCurve(X, Y, color="green", style="-", size=1)
#    X,Y = KarstenDOA_calculateCurve_linear(micC, micD, 3*tdoa*1/48000*343.2, res=0.1, rang=100)
#    drawCurve(X, Y, color="red", style="-", size=1)

finishDrawing(-100, -0, 100, 100, "Array "+array, "", "")