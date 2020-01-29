# -*- coding: utf-8 -*-


from GeometryLibrary import getPoint, distance, getPoint_on_tCurve, getClosestPoint, getPoint_on_tCurve_idx
import numpy as np

# Verbessere Schnittpunkt linearer Kurven durch finden des Schnittpunktes der Nichtlinearen Kurven
# >>Input:  der Schnittpunkt der linearen Kurven (linGoal), die beiden Kurven (curveA, curveB)
#           Parameter des Algorithmus (noch nicht durch Parameterliste gesteuert):
#           - maxIt                = Anzahl der maximalen Iterationen der Suche (def 50)
#           - startStepSize        = Start Schrittweite (def 1.0)
#           - intervallSearchMaxIt = maximale Iterationsanzahl bei Intervallsearch (def 50)
#           - epsilon              = Abbruchkriterium, wenn Abstand der Kurvenpunkte unter epsilon fällt, wird abgebrochen,
#                                    d.h. man nimmt vereinfachend an, der Schnittpunkt wurde schon gefunden (def 0.00001)
# >>Output: numerisch und iterativ ermittelter Schnittpunkt der beiden nichtlinearen Kurven
def optimizeIntersectionPoint_nonLinear_numeric(linGoal, curveA, curveB):
    # Initial Guess for t_A and t_B
       # hier am Anfang ohne max_iterations und großes startIntervall, damit Qualität der Ausgangswerte hoch
    startIntervall = getPoint(0,1000)
    endIntervall = intervallSearch_t_closest(intervall=startIntervall, curve=curveA, point=linGoal, iteration=0, max_iterations=100)
    tA = np.average(endIntervall)
    endIntervall = intervallSearch_t_closest(intervall=startIntervall, curve=curveB, point=linGoal, iteration=0, max_iterations=100)
    tB = np.average(endIntervall)
    
    # Determine indices
    p1, p2 = getPoint_on_tCurve(curveA, tA)
    p, iA = getClosestPoint(linGoal, [p1, p2])
    p1, p2 = getPoint_on_tCurve(curveB, tB)
    p, iB = getClosestPoint(linGoal, [p1, p2])
    
    # Optimize t_A and t_B iteratively
    # Parameters
    maxIt = 50
    startStepSize = 1.0
    intervallSearchMaxIt = 50
    epsilon = 0.00001
    # working variables
    it = 0
    stepSize = startStepSize
    pointA = getPoint_on_tCurve_idx(curveA, tA, iA)
    pointB = getPoint_on_tCurve_idx(curveB, tB, iB)
    oldDist = distance(pointA, pointB)
    while(oldDist>epsilon):
        tA, tB = gridSearch(tA, tB, curveA, curveB, iA, iB, stepSize=stepSize)
        endIntervall = intervallSearch_t_closest(intervall=getPoint(tB-stepSize,tB+stepSize), curve=curveB, point=getPoint_on_tCurve_idx(curveA, tA, iA), iteration=0, max_iterations=intervallSearchMaxIt)
        tB = np.average(endIntervall)

        pointA = getPoint_on_tCurve_idx(curveA, tA, iA)
        pointB = getPoint_on_tCurve_idx(curveB, tB, iB)
        
        newDist = distance(pointA,pointB)
        if(oldDist==newDist):
            stepSize = stepSize/2
        oldDist = newDist
        it+=1
        if(it>maxIt):
            break      
    return (pointA+pointB)/2

# Rastersuche für tA und tB auf zwei Kurven A und B, um Schnittpunkt der Kurvenäste zu bestimmen
# >>Input:  Startwerte für t Parameter (tA und tB), Kurven (curveA, curveB), Kurvenastindizes (iA, iB), Schrittweite (stepSize)
# >>Output: Beste gefundene Werte für tA und tB
def gridSearch(tA, tB, curveA, curveB, iA, iB, stepSize):
    f1 = np.arange(-stepSize,+stepSize,stepSize/10)
    f2 = np.arange(-stepSize,+stepSize,stepSize/10)
    value = np.linalg.norm(getPoint_on_tCurve_idx(curveA, tA, iA)-getPoint_on_tCurve_idx(curveB, tB, iB))
    ttA = tA
    ttB = tB
    for z1 in f1:
        for z2 in f2:
            tvalue = np.linalg.norm(getPoint_on_tCurve_idx(curveA, tA+z1, iA)-getPoint_on_tCurve_idx(curveB, tB+z2, iB))
            if(tvalue<value):
                value = tvalue
                ttA = tA+z1
                ttB = tB+z2
    return ttA, ttB
#def gridSearch(tA, tB, curveA, curveB, iA, iB, stepSize):
#    f = np.arange(-stepSize,+stepSize,stepSize/10)
#    value = np.linalg.norm(getPoint_on_tCurve_idx_new(curveA, tA, iA)-getPoint_on_tCurve_idx_new(curveB, tB, iB))
#    ttA = tA
#    ttB = tB
#    for z in f:
#        tvalue = np.linalg.norm(getPoint_on_tCurve_idx_new(curveA, tA+z, iA)-getPoint_on_tCurve_idx_new(curveB, tB+z, iB))
#        if(tvalue<value):
#            value = tvalue
#            ttA = tA+z
#            ttB = tB+z
#    return ttA, ttB
#    
# Intervallsuche für t auf Kurve, sodass Kurvenpunkt möglichst nahe an Zielpunkt (der nicht auf Kurve liegen muss)
# >>Input:  Ein Startintervall für t (intervall), eine Kurve (curve), ein Zielpunkt (point), die Iterationsnummer (iteration)
#           die Anzahl der maximalen Iterationen (max_iterations)
# >>Output: Das bestimmte Intervall für t nach der Suche
def intervallSearch_t_closest(intervall, curve, point, iteration, max_iterations):    
    pointA1, pointA2 = getPoint_on_tCurve(curve, intervall[0])
    pointC1, pointC2 = getPoint_on_tCurve(curve, intervall[1])
    listA = list()
    listA.append(pointA1)
    listA.append(pointA2)
    listC = list()
    listC.append(pointC1)
    listC.append(pointC2)
    pointA, ia = getClosestPoint(point,listA)
    pointC, ic = getClosestPoint(point,listC)
    valueA = np.linalg.norm(point-pointA)
    valueC = np.linalg.norm(point-pointC)
    if(valueA<valueC):  # it is in the Intervall A to B
        intervall = getPoint(intervall[0],np.average(intervall))
    else:
        intervall = getPoint(np.average(intervall),intervall[1])
    if(iteration<max_iterations):
        return intervallSearch_t_closest(intervall, curve, point, iteration+1, max_iterations)
    else:
        return intervall