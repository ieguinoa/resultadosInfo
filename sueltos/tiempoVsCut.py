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

#machePath = "/home/tati/Output_Mache/"
#sanderPath = "/home/tati/Output_Sander/"
#outputFile = open(sys.argv[1]+"_Mod", "w")

#dirs = [x for y in os.listdir(machePath) if os.path.isdir(sanderPath+x) and os.path.isdir(sanderPath+x)]
#dirs= ["500_100/"]
#print dirs

fig = plt.figure()

inputFile=open(sys.argv[1], "r")

cuts=[]
timesCAM=[]
timesCDM=[]
timesCTM=[]
timesDM=[]
timesDT=[]
timesTM=[]
timesTT=[]
timesAM=[]

prevCut=-1
for lines in inputFile:
  cut=int(lines.split()[0])
  if cut>prevCut:
    cuts.append(cut)
    prevCut=cut
  time=float(lines.split()[1])
  tipo=lines.split()[2]
  
  if tipo=="--cpu-analytic-memory":
      timesCAM.append(time)
  if tipo=="--cpu-tabla-memory":
      timesCTM.append(time)
  if tipo=="--cpu-deriv-memory":
      timesCDM.append(time)
  if tipo=="--analytic-memory":
      timesAM.append(time)
  if tipo=="--tabla-memory":
      timesTM.append(time) 
  if tipo=="--tabla-texture":
      timesTT.append(time)
  if tipo=="--deriv-memory":
     timesDM.append(time)
  if tipo=="--deriv-texture":
    timesDT.append(time)
    
    
    
np_cut = np.array(cuts)
    
#np_time = np.array(timesCAM)
#p1 = plt.plot(np_cut, np_time, lw=2, label="Analitico-CPU")

#np_time = np.array(timesCTM)
#p1 = plt.plot(np_cut, np_time, lw=2, label="TablaPot-CPU")

#np_time = np.array(timesCDM)
#p1 = plt.plot(np_cut, np_time, lw=2, label="TablaDeriv-CPU")

np_time = np.array(timesAM)
p1 = plt.plot(np_cut, np_time, lw=2, label="Analitico-GPU")

np_time = np.array(timesTM)
p1 = plt.plot(np_cut, np_time, lw=2, label="TablaPot-Memoria-GPU")

np_time = np.array(timesTT)
p1 = plt.plot(np_cut, np_time, lw=2, label="TablaPotTextura-GPU")

np_time = np.array(timesDM)
p1 = plt.plot(np_cut, np_time, lw=2, label="TablaDeriv-Memoria-GPU")

np_time = np.array(timesDT)
p1 = plt.plot(np_cut, np_time, lw=2, label="TablaDeriva-Textura-GPU")



legend=plt.legend(bbox_to_anchor=(0.2,0.8), loc=4, borderaxespad=0.,fancybox=True,shadow=True,title='Method')
plt.grid()
plt.show()
fig.savefig('out.png', dpi = 100)
