# -*- coding: utf-8 -*-

import numpy as np

# Generates a point
# >>Input:  X and Y coordinate of point
# >>Output: a point object (is np array type float64 shape (2,) )
def getPoint(x, y):
    return np.asarray([x, y])

# Determine euclidean distance between two points
# >>Input:  two points pointA and pointB
# >>Output: euclidean distance between two points
def distance(pointA, pointB):
    return np.linalg.norm(pointA-pointB)

# Get intersection points of two circles
# Coordinate system: pointA(0,0), horizontal axis through pointA and pointB
# >>Input:  Circle Radius: d1, d2, Distance between microphones: dm
# >>Output: a list of intersection points (can have 0,1 or 2 elements)
def getIntersectionPointsCircle_simple(d1, d2, dm):
    intersect_points = list()
    x        = (d1*d1-d2*d2+dm*dm)/(2*dm)
    sqrt_sub = d2*d2-(x-dm)*(x-dm)
    if(sqrt_sub>0):
        intersect_points.append(getPoint(x,+np.sqrt(sqrt_sub)))
        intersect_points.append(getPoint(x,-np.sqrt(sqrt_sub)))
    elif(sqrt_sub==0):
        intersect_points.append(getPoint(x,0))
    return intersect_points

# Get angle between two points (let pointA be (0,0), then angle=0° <=> pointB = (1,0)  )
# >>Input:  two points pointA, pointB
# >>Output: the angle between the two points, can only be between 0° and 360°!
def getAngle_angle1(pointA, pointB):
    dx = pointA[0]-pointB[0]
    dy = pointA[1]-pointB[1]
    if(dx==0):
        if(dy<0):
            return np.pi/2
        else:
            return -np.pi/2
    elif(dx<0):
        if(dy>0):
            return 2*np.pi+np.arctan(dy/dx)
        else:
            return np.arctan(dy/dx)
    else:
        return np.pi+np.arctan(dy/dx)
    
# Get intersection points of two circles
# >>Input:  Circle Radius: d1, d2, Microphone positions: pointA, pointB
# >>Output: a list of intersection points (can have 0,1 or 2 elements)
def getIntersectionPointsCircle(pointA, d1, pointB, d2):
    dm = distance(pointA, pointB)
    intersect_points = getIntersectionPointsCircle_simple(d1, d2, dm)
    if(len(intersect_points)==0):
        return list()
    
    # Determine Transformation Parameters
    angle0  = 0
    trans0  = getPoint(0, 0)
    trans1X = getPoint(intersect_points[0][0],0)
    angle2  = getAngle_angle1(pointA, pointB)
    transC  = getPoint(pointA[0]+np.cos(angle2)*trans1X[0], pointA[1] +np.sin(angle2)*trans1X[0])

    # Transform Points
    for i in range(0, len(intersect_points)):
        intersect_points[i] = getHomogCoord(intersect_points[i])
        intersect_points[i] = coordinateTransform(intersect_points[i], angle0, -trans1X)
        intersect_points[i] = coordinateTransform(intersect_points[i], angle2, +trans0)
        intersect_points[i] = coordinateTransform(intersect_points[i], angle0, +transC)
        intersect_points[i] = getCartesCoord(intersect_points[i])
    return intersect_points

# Converts angle from degree to radians
# >>Input:  angle in degree
# >>Output: angle in radians
def angle_radians(angle):
    return angle*(2*np.pi)/360

# Converts angle from radians to degree
# >>Input:  angle in radians
# >>Output: angle in degree
def angle_degree(angle):
    return angle*360/(2*np.pi)

# Converts point to homogenous coordinates
# >>Input:  point (2d = normal cartesian coordinates)
# >>Output: point (3d = 2d but in homogenous coordinates)
def getHomogCoord(point):
    return np.asarray([point[0],point[1],1])

# Converts point to homogenous coordinates
# >>Input:  point (3d = 2d but in homogenous coordinates)
# >>Output: point (2d = normal cartesian coordinates)
def getCartesCoord(pointHom):
    return np.asarray([pointHom[0],pointHom[1]])

# Transforms point by rotation and translation
# >>Input:  point (3d = 2d but in homogenous coordinates)
# >>Output: transformed point (3d = 2d but in homogenous coordinates)
def coordinateTransform(point, rot_alpha, translation_vector):
    matrix = np.asarray([[np.cos(rot_alpha),-np.sin(rot_alpha),translation_vector[0]],[np.sin(rot_alpha),np.cos(rot_alpha),translation_vector[1]],[0,0,1]])
    return matrix.dot(point)

# Calculate Microfone Array 2 (eight microphones showing upwards)
# >>Input:  array_radius, points:  array_center 
# >>Output: liste mit microphone positionen (micList)
def calculateMicrophoneArray_2(array_radius, array_center):
    micList = list()
    for i in range(0,8):
        micList.append(getPoint(array_center[0]+array_radius*np.sin(2*np.pi/8*i),array_center[1]+array_radius*np.cos(2*np.pi/8*i)))
    return micList

# Calculate the nonlinear curve of possible soud source positions [ assumption Mic1 at (-dm/2,0), Mic2 at (+dm/2,0) ]
# (According to Karsten Kreutz Masterthesis)
# >>Input:  distance between microphones (dm), delta s (ds), resolution of curve (res), rang = range of curve
# >>Output: X and Y array that can be plotted, e.g. with plt.plot(X,Y)
def KarstenDOA_calculateCurve_nonlinear_simple(dm, ds, res, rang=10):
    # Create X Array
    if(ds==0):
        X = np.zeros_like(np.arange(-rang,+rang,res))
        Y = np.arange(-rang,+rang,res)
        return X,Y
    elif(ds>0):
        X1 = np.flip(np.arange(+0,+rang,res))
        X2 = np.arange(+0,+rang,res)
        X = np.concatenate((X1,X2))
    else:
        X1 = np.arange(-rang,0,res)
        X2 = np.flip(np.arange(-rang,+0,res))
        X = np.concatenate((X1,X2))
    # Remove all invalid X (for that the curve does not exist)
    X_list = list(X)
    X_del  = list()
    for i in range(0,len(X_list)):
        if((X[i]*X[i]-ds*ds/4)<0):
            X_del.append(i)
    for index in sorted(X_del, reverse=True):
        del X_list[index] 
    # Create Y Array
    X = np.asarray(X_list)
    Y = np.zeros_like(X)
    for i in range(0,X.shape[0]):
        if(i<X.shape[0]/2):
            Y[i] = -np.sqrt(    (X[i]*X[i]-ds*ds/4)  *  (dm*dm/(ds*ds)-1)     )
        else:
            Y[i] = +np.sqrt(    (X[i]*X[i]-ds*ds/4)  *  (dm*dm/(ds*ds)-1)     )
    return X, Y

# Calculate the linear approximation of curve of possible soud source positions [ assumption Mic1 at (-dm/2,0), Mic2 at (+dm/2,0) ]
# >>Input:  distance between microphones (dm), delta s (ds), resolution of curve (res), rang = range of curve
# >>Output: X and Y array that can be plotted, e.g. with plt.plot(X,Y)
def KarstenDOA_calculateCurve_linear_simple(dm, ds, res, rang=10):
    # Create X Array
    if(ds==0):
        X = np.zeros_like(np.arange(-rang,+rang,res))
        Y = np.arange(-rang,+rang,res)
        return X,Y
    if(ds>0):
        X1 = np.flip(np.arange(+0,+rang,res))
        X2 = np.arange(+0,+rang,res)
        X = np.concatenate((X1,X2))
    else:
        X1 = np.arange(-rang,0,res)
        X2 = np.flip(np.arange(-rang,+0,res))
        X = np.concatenate((X1,X2))
    # Calculate Y
    Y = np.zeros_like(X)
    for i in range(0,X.shape[0]):
        if(i<X.shape[0]/2):
            Y[i] = -X[i]*np.sqrt(dm*dm/(ds*ds)-1)
        else:
            Y[i] = +X[i]*np.sqrt(dm*dm/(ds*ds)-1)
    return X,Y

# Calculate the DOA angle [ assumption Mic1 at (-dm/2,0), Mic2 at (+dm/2,0) ]
# >>Input:  distance between microphones (dm), delta s (ds)
# >>Output: angle (can be between -PI/2 and + PI/2, if ds = 0, SS between Mics, angle is zero)
def KarstenDOA_calculateAngle_linear_simple(dm, ds):
    if(ds==0):
        return 0
    elif(dm*dm/(ds*ds)-1<0):
        return 0
    if(ds>0):
        m = +np.sqrt(dm*dm/(ds*ds)-1)
        alpha = np.pi/2-np.arctan(m)
    else :
        m = -np.sqrt(dm*dm/(ds*ds)-1)
        alpha = -(np.pi/2+np.arctan(m))
    return alpha

# Calculate the DOA angle [ assumption Mic1 at (-dm/2,0), Mic2 at (+dm/2,0) ]
# >>Input:  distance between microphones (dm), delta s (ds)
# >>Output: angle (can be between -PI/2 and + PI/2, if ds = 0, SS between Mics, angle is zero)
def KarstenDOA_calculateSteep_linear_simple(dm, ds):
    if(ds==0):
        return np.inf
    elif(dm*dm/(ds*ds)-1<0):
        return np.inf
    if(ds>0):
        m = +np.sqrt(dm*dm/(ds*ds)-1)
    else :
        m = -np.sqrt(dm*dm/(ds*ds)-1)
    return m

# Calculate the nonlinear curve of possible soud source positions
# (According to Karsten Kreutz Masterthesis)
# >>Input:  microphone positions 1 and 2, delta s (ds), resolution of curve (res), rang = range of curve
# >>Output: X and Y array that can be plotted, e.g. with plt.plot(X,Y)
def KarstenDOA_calculateCurve_nonlinear(mic1, mic2, ds, res, rang=10):
    dm = distance(mic1,mic2)
    X,Y = KarstenDOA_calculateCurve_nonlinear_simple(dm, ds, res, rang)       
    # Determine Transformation Parameters
    angle0  = 0
    trans0  = getPoint(0, 0)
    transCenter = getPoint((mic1[0]+mic2[0])/2,(mic1[1]+mic2[1])/2)
    angle2  = getAngle_angle1(mic1, mic2)
    # Transform Points
    for i in range(0, X.shape[0]):
        p = getPoint(X[i], Y[i])
        p = getHomogCoord(p)
        p = coordinateTransform(p, angle2, +trans0)
        p = coordinateTransform(p, angle0, +transCenter)
        p = getCartesCoord(p)    
        X[i] = p[0]
        Y[i] = p[1]
    return X, Y

# Calculate the linear approximation of curve of possible soud source positions
# >>Input:  microphone positions 1 and 2, delta s (ds), resolution of curve (res), rang = range of curve
# >>Output: X and Y array that can be plotted, e.g. with plt.plot(X,Y)
def KarstenDOA_calculateCurve_linear(mic1, mic2, ds, res, rang=10):
    dm = distance(mic1,mic2)
    X,Y = KarstenDOA_calculateCurve_linear_simple(dm, ds, res, rang)
    # Determine Transformation Parameters
    angle0  = 0
    trans0  = getPoint(0, 0)
    transCenter = getPoint((mic1[0]+mic2[0])/2,(mic1[1]+mic2[1])/2)
    angle2  = getAngle_angle1(mic1, mic2)
    # Transform Points
    for i in range(0, X.shape[0]):
        p = getPoint(X[i], Y[i])
        p = getHomogCoord(p)
        p = coordinateTransform(p, angle2, +trans0)
        p = coordinateTransform(p, angle0, +transCenter)
        p = getCartesCoord(p)    
        X[i] = p[0]
        Y[i] = p[1]
    return X, Y

# Get Intersectionpoints of two microphone pairs doa estimations for linear model (there are 4 possible points)
# >>Input:  Position of first pair (micA, micB) and second pair (micC, micD),  and DOA estimations of pairs (m1T and m2T, steepness)
# >>Output: a list of all intersection points,  and parameters of 4 different straights (Y = m_i*X + b_i)
def getMicrophonePair_DOA_Intersection_linear(micA, micB, micC, micD, m1T, m2T):
    p1 = (micA+micB)/2
    p2 = (micC+micD)/2
      
    angle1_a = +getAngle_angle1(micA, micB)
    angle1_b = -getAngle_angle1(micA, micB)
    angle2_a = +getAngle_angle1(micC, micD)
    angle2_b = -getAngle_angle1(micC, micD)
    
    if(m1T!=np.inf):
        m1_a = +(np.sin(angle1_a)+m1T*np.cos(angle1_a))/(np.cos(angle1_a)-m1T*np.sin(angle1_a))
        m1_b = -(np.sin(angle1_b)+m1T*np.cos(angle1_b))/(np.cos(angle1_b)-m1T*np.sin(angle1_b))
    else:
        m1_a = np.tan(np.pi/2-angle1_a)
        m1_b = np.tan(np.pi/2-angle1_b)
    b1_a = p1[1]-m1_a*p1[0]
    b1_b = p1[1]-m1_b*p1[0]
    
    if(m2T!=np.inf):
        m2_a = +(np.sin(angle2_a)+m2T*np.cos(angle2_a))/(np.cos(angle2_a)-m2T*np.sin(angle2_a))
        m2_b = -(np.sin(angle2_b)+m2T*np.cos(angle2_b))/(np.cos(angle2_b)-m2T*np.sin(angle2_b))
    else:
        m2_a = np.tan(np.pi/2-angle2_a)
        m2_b = np.tan(np.pi/2-angle2_b)
    b2_a = p2[1]-m2_a*p2[0]
    b2_b = p2[1]-m2_b*p2[0]
    
    pointList = list()
    if(m1_a!=np.nan):
        if(m2_a!=np.nan):
            pointS1 = getStraightIntersection(b1_a, b2_a, m1_a, m2_a)
            if(pointS1[0]!=np.nan):
                pointList.append(pointS1)
        if(m2_b!=np.nan):
            pointS2 = getStraightIntersection(b1_a, b2_b, m1_a, m2_b)
            if(pointS2[0]!=np.nan):
                pointList.append(pointS2)
    if(m1_b!=np.nan):
        if(m2_a!=np.nan):
            pointS3 = getStraightIntersection(b1_b, b2_a, m1_b, m2_a)
            if(pointS3[0]!=np.nan):
                pointList.append(pointS3)
        if(m2_b!=np.nan):
            pointS4 = getStraightIntersection(b1_b, b2_b, m1_b, m2_b)
            if(pointS4[0]!=np.nan):
                pointList.append(pointS4)
                
    return pointList, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b
     
# Get intersection point of two straight lines
# >>Input:  Straight i: Y = m_i*X + b_i,   for straight 1 and 2
# >>Output: the point of intersection
def getStraightIntersection(b1, b2, m1, m2):
    if(m1-m2==0):
        return getPoint(np.nan, np.nan)
    else:
        x = (b2-b1)/(m1-m2)
        y = m1*x+b1
        return getPoint(x, y)
    
# get the closest point in a list to a given (goal) point
# >>Input:  the given point (goal), and a list of points (pointList)
# >>Output: the closes point in pointList (goal) and the index of this point in the list
def getClosestPoint(goal, pointList):
    minD = np.inf
    indD = -1
    for p in range(0,len(pointList)):
        dist = np.linalg.norm(pointList[p]-goal)
        if(dist<minD):
            indD = p
            minD = dist
    goal = pointList[indD]    
    return goal, indD
    
# Erstelle eine t-parametrisierte Kurve eines Mikrofonpaares
# >>Input:  Mikrofon Position mic1 und mic2, delta s ds
# >>Output: tCurve Objekt
def get_tCurve(mic1, mic2, ds):
    tCurve = {}
    tCurve["ds"]   = ds
    if(ds>0):
        tCurve["mic1"] = mic1
        tCurve["mic2"] = mic2
    else:
        tCurve["mic1"] = mic2
        tCurve["mic2"] = mic1
    return tCurve

# Ermittelt Punkte auf einer t-parametrisierten Kurve bei gegebenem t
# >>Input:   Kurve tCurve, t Wert
# >> Output: Zwei mögliche Punkte (je nach Kurvenast)
def getPoint_on_tCurve(tCurve, t):
    ds    = tCurve["ds"]
    P     = (tCurve["mic1"]+tCurve["mic2"])/2
    dm    = distance(tCurve["mic1"],tCurve["mic2"])
    angle = getAngle_angle1(tCurve["mic1"], tCurve["mic2"])
    if(ds!=0):
        X1 = np.cos(angle)*(t+ds/2)  - np.sin(angle)*(+ np.sqrt(        (np.power(t+ds/2,2) - ds*ds/4) * (dm*dm/(ds*ds)-1)       )) + P[0]
        X2 = np.cos(angle)*(t+ds/2)  - np.sin(angle)*(- np.sqrt(        (np.power(t+ds/2,2) - ds*ds/4) * (dm*dm/(ds*ds)-1)       )) + P[0]
        Y1 = np.sin(angle)*(t+ds/2)  + np.cos(angle)*(+ np.sqrt(        (np.power(t+ds/2,2) - ds*ds/4) * (dm*dm/(ds*ds)-1)       )) + P[1]
        Y2 = np.sin(angle)*(t+ds/2)  + np.cos(angle)*(- np.sqrt(        (np.power(t+ds/2,2) - ds*ds/4) * (dm*dm/(ds*ds)-1)       )) + P[1]
        return getPoint(X1,Y1), getPoint(X2,Y2)
    else:
        X1 = np.cos(angle)*P[0] - np.sin(angle)*(P[1]+t)
        X2 = np.cos(angle)*P[0] - np.sin(angle)*(P[1]-t)
        Y1 = np.sin(angle)*P[0] + np.cos(angle)*(P[1]+t)
        Y2 = np.sin(angle)*P[0] + np.cos(angle)*(P[1]-t)
        return getPoint(X1,Y1), getPoint(X2,Y2)

# Ermittelt einen Punkt auf einer t-parametrisierten Kurve bei gegebenem t
# >>Input:   Kurve tCurve, t Wert, Kurvenast idx
# >> Output: Ein Punkt (je nachdem welcher Kurvenast durch idx gewählt ist)
def getPoint_on_tCurve_idx(tCurve, t, idx):
    p1, p2 = getPoint_on_tCurve(tCurve, t)
    if(idx==0):
        return p1
    else:
        return p2    


def getAngle_Pair(micA_pos, micB_pos, ds):
    dm = distance(micA_pos, micB_pos)
    return KarstenDOA_calculateAngle_linear_simple(dm, ds)  

def estimateK_Pair(powerA, powerB, micA_pos, micB_pos, ds):
    alpha = np.pi/2- getAngle_Pair(micA_pos, micB_pos, ds)    
    amplitude_A = np.sqrt(powerA)
    amplitude_B = np.sqrt(powerB)
    
    # Calculate K
    dm = distance(micA_pos, micB_pos)
    ampDiff = 1/(amplitude_A*amplitude_A) - 1/(amplitude_B*amplitude_B)
    ang = np.sqrt(np.tan(alpha)*np.tan(alpha)+1)
    left = dm / (ampDiff*amplitude_A*ang)
    right = np.sqrt( (dm*dm)/(amplitude_A*amplitude_A*ang*ang*ampDiff*ampDiff)    -    (dm*dm)/(ampDiff)    )

    K1 = np.abs(left + right)
    K2 = np.abs(left - right)
    return K1*K1, K2*K2

# =============================================================================
# # DEMO: a simple animated geometry drawing demo
# =============================================================================
#from GraphicLibrary import initDrawing, drawCircle, drawPoint, drawLine, drawString
#import matplotlib.pyplot as plt
#initDrawing()
#plt.title("DemoGeometry_1.py")
#plt.xlabel("X-Axis")
#plt.ylabel("Y-Axis")
#radius = 2
#ss     = getPoint(2, 2)
#
#for x in range(0,201):
#    plt.clf()
#    plt.title("DemoGeometry_1.py")
#    plt.xlabel("X-Axis")
#    plt.ylabel("Y-Axis")
#    plt.xlim(-4,4)
#    plt.ylim(-4,4)
#    plt.grid()
#    # Determine angle
#    angle = np.pi*2/100*x
#    
#    # Define Geoemtric Situation
#    mic1 = getPoint(0, 0)
#    mic2 = getPoint(radius*np.cos(angle), radius*np.sin(angle))
#    
#    # Here comes signal processing
#    d1 = distance(mic1, ss)
#    d2 = distance(mic2, ss)
#    
#    # Calculate stuff
#    intersect_points = getIntersectionPointsCircle(mic1, d1, mic2, d2)
#    
#    # Draw Stuff
#    drawPoint(mic1, "x", "blue", 50)
#    drawPoint(mic2, "x", "red",  50)
#    
#    drawLine(mic1, mic2, "-", "black", 1)
#    drawLine(mic1, ss, ":", "red", 1)
#    drawLine(mic2, ss, ":", "blue", 1)
#    
#    drawString("Mic1", mic1+getPoint(-0.2,0.2), "blue", 15)
#    drawString("Mic2", mic2+getPoint(-0.2,0.2), "red", 15)
#    drawString("Angle1: "+'%.2f' % angle_degree(getAngle_angle1(mic1,mic2))+"°", getPoint(-2,3), "black", 15)
#    
#    drawCircle(mic1, d1, "red")
#    drawCircle(mic2, d2, "blue")
#
#    for i in range(0,len(intersect_points)):
#        drawPoint(intersect_points[i], "o", "black", 50)
#    plt.pause(0.05)

