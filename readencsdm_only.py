import os
import pyfits
import numpy as np
import matplotlib
matplotlib.use('agg')
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import math
import h5py
import re
import sys
import glob
from numpy.linalg import inv
rcParams['figure.figsize'] = 8, 6
rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 24})
rcParams['axes.unicode_minus'] = False
import matplotlib.patches as patches
rcParams['axes.linewidth'] = 2
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
from readsnap_samson import *
from Sasha_functions import *
from gadget_lib.cosmo import *
from samson_functions import *
from enclosedmass import enclosedmass
#dirneed=['m10h1146','m10h573','m10h553','m10h476','m11h383','m11']
#dirneed=['f1146','f573','f553','f476','f383','fm11']
dirneed=[['f573',8.2,13.5],['f553',5.7,13.5],['f476',2.6,13.5],['fm11',2.3,6.0],['f383',2.0,6.5],['f61',2.2,2.4]]
#dirneed=['m11']
#dirneed=['553','476']
maxr=20.0
minr=0.5
galcen=1
#Nsnap=440

for runtodol in dirneed:
	runtodo=runtodol[0]
	time = runtodol[1]
	emdata = enclosedmass(runtodo, time,minr,maxr, galcen=galcen)
	rlist =  emdata['rlist']
	vollist = emdata['vollist']
	Gmlist = emdata['Gmlist']
	Smlist = emdata['Smlist']
	DMmlist = emdata['DMmlist']
	haloinfo=cosmichalo(runtodo)
	labelname=haloinfo['labelname']
	halocolor=haloinfo['halocolor']
        print 'halocolor', halocolor
	print 'labelname', labelname
	plt.plot(rlist,Smlist+DMmlist, color=halocolor,label=labelname)
	time = runtodol[2]
        emdata = enclosedmass(runtodo, time,minr,maxr, galcen=galcen)
        rlist =  emdata['rlist']
        vollist = emdata['vollist']
        Gmlist = emdata['Gmlist']
        Smlist = emdata['Smlist']
        DMmlist = emdata['DMmlist']
        plt.plot(rlist,Smlist+DMmlist,ls='--', color=halocolor)
plt.errorbar([2.4,8.1], [2.6e9,4.5e9], yerr=[[1.3e9,3.4e9],[2.8e9,2.8e9]],label='VCC1287', fmt='o')
plt.errorbar(4.7, 7.0e9, xerr=[[0.2],[0.2]], yerr=[[2.0e9],[3.0e9]],label='Dragonfly44',  fmt='s')
#plt.errorbar(5.0, 4.6e9, yerr=[[0.8e9],[0.8e9]],label='UGC2162',  fmt='^')
plt.xscale('log')
plt.yscale('log')
plt.xlim([0.5,20])
plt.xlabel(r'$r\; ({\rm kpc})$')
plt.ylabel(r'${\rm M}_{\rm enc} ({\rm M}_{\odot})$')
plt.legend(loc='best',fontsize=12)
plt.savefig('figures/Mencdm_udg_sdmonly.pdf')
plt.clf()
