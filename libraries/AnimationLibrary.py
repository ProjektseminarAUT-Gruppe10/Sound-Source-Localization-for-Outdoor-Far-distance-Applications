
# -*- coding: utf-8 -*-
import os
import imageio
import matplotlib.pyplot as plt

# Creates a screenshot of a given pyplot and stores it into a file (e.g. for gif creation)
# >>Input:  path = path to folder to store picture, number = id of picture in strip, dpi = details per inch
# >>Output: (none), but it will create a png-file
def screenshot(path, number, dpi):
    plt.gcf().savefig(path+"/pic_"+"{0:0=3d}".format(number)+'.png', dpi=100)

# Generate a list of all files within a given folder
# >>Input:  path = path to folder with files
# >>Output: list with all filenames ("path/filename.filetype")
def listFiles(path):
    fList = list() 
    for root, dirs, files in os.walk(path):  
        for filename in files:
            fList.append(path+"/"+filename)
    fList.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))              
    return fList 

# Generate a gif from images
# >>Input:  pathA = path to folder with pictures, pathB = destination file for gif, delay = speed of gif
# >>Output: (none), but it will create a gif-file
def generateGif(pathA,pathB,delay):
    fileList = listFiles(pathA)
    with imageio.get_writer(pathB, mode='I') as writer:
        for filename in fileList:
            image = imageio.imread(filename)
            writer.append_data(image)
        for i in range(0,delay):
            image = imageio.imread(fileList[len(fileList)-1])
            writer.append_data(image) 