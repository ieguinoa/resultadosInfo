#!/usr/bin/python

import operator
import os
from math import *
import sys
from os.path import isfile
from itertools import *
import numpy as np
import matplotlib.mlab as mlab
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

import pylab as plot
params = {'legend.fontsize': 20,
          'legend.linewidth': 4}
plot.rcParams.update(params)

machePath = "/home/tati/Output_Mache/"
sanderPath = "/home/tati/Output_Sander/"


print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)


if sys.argv[2] == "2":
  #imprimir la segunda columna
  outputFile = open(sys.argv[1]+"_Mod", "w")
  inputFile = open(sys.argv[1])
  for l in inputFile:
	    time=l.split()[1]
	    outputFile.write(time + "\n")
	    #steps.append(int(l.split()[0])/4)
	    #difValor= -1148.8685822590464 - float(l.split()[1])
	    ##difValor= -8.6741 - float(l.split()[1])
	    #valorCuadrado= pow(difValor,2)
	    #rmsd=sqrt(valorCuadrado)
	    #time.append(rmsd)
	    ##time.append(float(l.split()[1]))
	  #inputFile.close()
else:
  outputFile = open(sys.argv[1]+"_Mod1", "w")
  inputFile = open(sys.argv[1])
  for l in inputFile:
	    time=l.split()[0]
	    outputFile.write(time + "\n")


##dirs = [x for y in os.listdir(machePath) if os.path.isdir(sanderPath+x) and os.path.isdir(sanderPath+x)]
#dirs= ["500_100/"]
#print dirs
#fig = plt.figure()
#for d in dirs:
    #for f in os.listdir(machePath+d):
      #if f.endswith(".out") and f.startswith("time"):
	#print f
	#inputFile = open(machePath + d + "/" + f)
	#steps = []      
	#time = []
	
	#for l in inputFile:
	  #steps.append(int(l.split()[0])/4)
	  #difValor= -1148.8685822590464 - float(l.split()[1])
	  ##difValor= -8.6741 - float(l.split()[1])
	  #valorCuadrado= pow(difValor,2)
	  #rmsd=sqrt(valorCuadrado)
	  #time.append(rmsd)
	  ##time.append(float(l.split()[1]))
	#inputFile.close()
	
	#np_time = np.array(time)
	#np_step = np.array(steps)
	#p1 = plt.plot(np_step, np_time, lw=2, label="ErrorFuerzas_"+ f)
    
    #for f in os.listdir(sanderPath+d):
      #if f.endswith(".out") and f.startswith("time"):
	#print f
	#inputFile = open(sanderPath + d + "/" + f)
	#steps = []      
	#time = []
	
	#for l in inputFile:
	  #stp = int(l.split()[0])
	  #steps.append(stp)
	  #time.append(float(l.split()[1]))		#Time is in S and is an average
	#inputFile.close()
	
	#np_time = np.array(time)
	#np_step = np.array(steps)
	#p1 = plt.plot(np_step, np_time, lw=2, label="snd_"+f)

    #legend=plt.legend(bbox_to_anchor=(0.8,0.4), loc=4, borderaxespad=0.,
			#fancybox=True,shadow=True,title='Method')
    #plt.grid()
    #plt.show()
    #fig.savefig(machePath+d+'/out.png', dpi = 100)
