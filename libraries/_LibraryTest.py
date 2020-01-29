# Imports
from GeometryLibrary import getPoint, distance
from SslLibrary import SSL_TDOA_LN, SSL_TDOA_NL, SSL_AMPL
from GraphicLibrary import drawPoint, drawPointL, initDrawing, finishDrawingL
from MicrophonePositionLibrary import getMicrophonePositions_SIM_A, getMicrophonePositions_SIM_B, getMicrophonePositions_SIM_C, getMicrophonePositions4_TDOA, getMicrophonePositions4_AMP

# Define Physics
c_speed = 300

# Define Microphones
micPosList = getMicrophonePositions_SIM_C(0.3)
micPosList_TDOA = getMicrophonePositions4_TDOA(micPosList)
micPosList_AMP  = getMicrophonePositions4_AMP(micPosList)

# Define Soundsource
soundSource = getPoint(10, 5)

# Signal Processing which is fake
tdoa1 = (1/c_speed) * (distance(micPosList_TDOA[0], soundSource) - distance(micPosList_TDOA[1], soundSource))
tdoa2 = (1/c_speed) * (distance(micPosList_TDOA[2], soundSource) - distance(micPosList_TDOA[3], soundSource))

distA = distance(micPosList_AMP[0], soundSource)
distB = distance(micPosList_AMP[1], soundSource)

# Calculate Estimations
point1 = SSL_TDOA_LN(micPosList_TDOA, tdoa1, tdoa2, c_speed)
point2 = SSL_TDOA_NL(micPosList_TDOA, tdoa1, tdoa2, c_speed, point1)
point3 = SSL_AMPL(micPosList_AMP, distA, distB)

# View Results
fig, ax = initDrawing()

# Draw Microphones
for p in micPosList:
    drawPoint(p, "o", "black", 50)  
drawPointL(micPosList[0], "o", "black", 50, "Microphones")  

# Draw SoundSource position
drawPointL(soundSource, "v", "blue", 100, "Sound Source")

# Draw Estimated Locations
drawPointL(point1, "v", "red", 50, "SSL_TDOA_LN")
drawPointL(point2, "v", "green", 25, "SSL_TDOA_NL")
drawPointL(point3, "v", "yellow", 10, "SSL_AMPL")

# Finish Drawing
finishDrawingL(-40, -5, +40, +45, "Comparison of SSL Algorithms", "X-Axis", "Y-Axis")
