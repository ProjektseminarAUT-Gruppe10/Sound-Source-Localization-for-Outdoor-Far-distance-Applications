# -*- coding: utf-8 -*-

#imports
from GeometryLibrary import KarstenDOA_calculateSteep_linear_simple, distance, getMicrophonePair_DOA_Intersection_linear, get_tCurve, getIntersectionPointsCircle
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric

# Localize Sound Source using TDOA (and linear intersection points)
# >>Input:  four points within micPosList, tdoa1 and 2 for the two pairs, the speed of sound
# >>Output: a point where the sound source localization should be
def SSL_TDOA_LN(micPosList, tdoa1, tdoa2, c_speed):
    # Get Microphone Points
    micAP = micPosList[0]
    micBP = micPosList[1]
    micCP = micPosList[2]
    micDP = micPosList[3]
    average = (micAP+micBP+micCP+micDP)/4
    
    # Calculate way_tdoa1 and way_tdoa2
    ds1 = tdoa1*c_speed
    ds2 = tdoa2*c_speed
    
    # Calculate Linear IntersectionPoints
    m1T = KarstenDOA_calculateSteep_linear_simple(distance(micAP, micBP), ds1)
    m2T = KarstenDOA_calculateSteep_linear_simple(distance(micCP, micDP), ds2)
    pointList, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micAP, micBP, micCP, micDP, m1T, m2T)

    # Choose Plausible One = the one that is farest away from microphones
    pointListC = list()
    for p in pointList:
        if(p[1]>0):
            pointListC.append(p)
    if(distance(pointListC[0],average)<distance(pointListC[1],average)):
        return pointListC[1]
    else:
        return pointListC[0]

# Localize Sound Source using TDOA (and nonlinear intersection points)
# >>Input:  four points within micPosList, tdoa1 and 2 for the two pairs, the speed of sound, and the linear intersection point as initial value
# >>Output: a point where the sound source localization should be
def SSL_TDOA_NL(micPosList, tdoa1, tdoa2, c_speed, LIN):
    # Get Microphone Points
    micAP = micPosList[0]
    micBP = micPosList[1]
    micCP = micPosList[2]
    micDP = micPosList[3]
    
    # Calculate way_tdoa1 and way_tdoa2
    ds1 = tdoa1*c_speed
    ds2 = tdoa2*c_speed

    # Define Curves
    curveA = get_tCurve(micAP, micBP, ds1)
    curveB = get_tCurve(micCP, micDP, ds2)

    # Find Intersection Point of Nonlinear Curves
    return optimizeIntersectionPoint_nonLinear_numeric(LIN, curveA, curveB)

# Localize Sound Source using Amplitude (intersection points of circles)
# >>Input:  two points within micPosList, estim distances of microphones from signal
# >>Output: a point where the sound source localization should be
def SSL_AMPL(micPosList, distA, distB):
    # Get Microphone Points
    micAP = micPosList[0]
    micBP = micPosList[1]
    average = (micAP+micBP)/2

    # Find Intersection Points of microphones
    intersect_points = getIntersectionPointsCircle(micAP, distA, micBP, distB)
    
    # Choose Plausible One = the one that is farest away from microphones
    if(distance(intersect_points[0],average)<distance(intersect_points[1],average)):
        return intersect_points[1]
    else:
        return intersect_points[0]
#    if(intersect_points[0][1]<0 and intersect_points[1][1]>0):
#        RES = intersect_points[1]
#    elif(intersect_points[1][1]<0 and intersect_points[0][1]>0):
#        RES = intersect_points[0]
#    else:
#        average = (micAP+micBP)/2
#        if(distance(average,intersect_points[0])>distance(average,intersect_points[1])):
#            RES = intersect_points[0]
#        else:
#            RES = intersect_points[1]
#    return RES