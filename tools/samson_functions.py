from struct import *
import sys
import os
import matplotlib.pyplot as plt
import pylab
import numpy as np
import math
from distcalcs import *
from zip_em_all import *
import time
from scipy.interpolate import interp1d
from datetime import date
import scipy.stats as stats
import scipy.optimize as optimize
import errno
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from lum_mag_conversions import luminosity_to_magnitude
mpl.use('Agg')
from Sasha_functions import read_halo_history
from readsnap_samson import *

pi  = np.pi
sin = np.sin
cos = np.cos

def ellipse(u,v):
    x = rx*cos(u)*cos(v)
    y = ry*sin(u)*cos(v)
    z = rz*sin(v)
    return x,y,z


def mvee(points, tol = 0.001):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c,c))/d
    return A, c


# This routine uses relative positions of the particles to locate the center. The basic mechanism is to group particles into rectangular grids, and then fit the high density grid with ellipsoidal. 

#DMrelpos is the positions of the particles and nobin^3 is the number of grid. 

#nopafactor* den is the density cutoff (den is the average density of the particles). 

# !It is better to keep nobin <100 or the code will cost a lot of time and the center will be locating the highest density clump but not based on overall shape 

# ! It is not possible to fit with too many or too few grids (if nopa is too small, the number of grid will be too large and you should tune the nopafactor


def ellipsoidal_centering(DMrelposX,DMrelposY,DMrelposZ,nobin,nopafactor):
	DMco=np.array([DMrelposX,DMrelposY,DMrelposZ])
	DMcor=DMco.T

	DMparno=len(DMrelposX)
	H, edges = np.histogramdd(DMcor, bins = (nobin, nobin, nobin))

	NominS=[]
	ellX=[]
	ellY=[]
	ellZ=[]

	den=float(DMparno)/(pi*4./3.)/float(nobin)/float(nobin)/float(nobin)

	nopa=float(den)*nopafactor

	Dpos=[]
	Dposx=[]
	Dposy=[]
	Dposz=[]
	totalno=0
	for i in range(nobin):
		for j in range(nobin):
			for k in range(nobin):
				if (H[i,j,k]>nopa):
					Dposx=np.append(Dposx,edges[0][i])
					Dposy=np.append(Dposy,edges[1][j])
					Dposz=np.append(Dposz,edges[2][k])
					totalno+=H[i,j,k]
	if len(Dposx)<4:
		print 'Warning: Density threshold is too high'
		return -1, -1, -1 
	if len(Dposx)>1000:
		print 'Warning: Density threshold is too low and the no of grids is too large'
		return -1, -1, -1

	print 'fitting ell'
	points = np.vstack(((Dposx),(Dposy),(Dposz))).T
	A, centroid = mvee(points)
	DXi=centroid[0]
	DYi=centroid[1]
	DZi=centroid[2]
	return DXi, DYi, DZi

def outdirname(runtodo, Nsnap=500):
        subdir='hdf5'
        timestep=''
        maindir='scratch'
        cosmo=0
	color='k'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test16c_bridges'):
                rundir='mw_cr_lr_dc28_1_23_17_test16c_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
		color='b'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test16chole_bridges'):
                rundir='mw_cr_lr_dc28_1_23_17_test16chole_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
		color='g'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test16cout_bridges'):
                rundir='mw_cr_lr_dc28_1_23_17_test16cout_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
		color='r'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_testhole'):
                rundir='mw_cr_lr_dc28_1_23_17_testhole'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc29_1_23_17_test6'):
                rundir='mw_cr_lr_dc29_1_23_17_test6'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e29'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
		color='b'
        if (runtodo=='mw_cr_lr_dc29_1_23_17_test6chole_bridges'):
                rundir='mw_cr_lr_dc29_1_23_17_test6chole_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e29'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
		color='g'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test6'):
                rundir='mw_cr_lr_dc28_1_23_17_test6'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_3_31_M1'):
                rundir='mw_cr_lr_dc28_3_31_M1'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_4_19_M1'):
                rundir='mw_cr_lr_dc28_4_19_M1'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_4_19_M1_equaltimestep'):
                rundir='mw_cr_lr_dc28_4_19_M1_equaltimestep'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc29_4_19_M1'):
                rundir='mw_cr_lr_dc29_4_19_M1'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e29'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_3_31_M1_hole'):
                rundir='mw_cr_lr_dc28_3_31_M1_hole'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_3_31_M1_equaltimestep'):
                rundir='mw_cr_lr_dc28_3_31_M1_equaltimestep'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_3_31_M1_equaltimestep_b4necheck'):
                rundir='mw_cr_lr_dc28_3_31_M1_equaltimestep_b4necheck'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_3_31_M1_equaltimestep_addboundpre'):
                rundir='mw_cr_lr_dc28_3_31_M1_equaltimestep_addboundpre'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_3_31_M1_equaltimestep_outcr'):
                rundir='mw_cr_lr_dc28_3_31_M1_equaltimestep_outcr'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_mr_dc28_1_23_17_test6'):
                rundir='mw_cr_mr_dc28_1_23_17_test6'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='mr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_testhole'):
                rundir='mw_cr_lr_dc28_1_23_17_testhole'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test6_noIa_bridges'):
                rundir='mw_cr_lr_dc28_1_23_17_test6_noIa_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test6hole_noIa_30Gyr_bridges'):
                rundir='mw_cr_lr_dc28_1_23_17_test6hole_noIa_30Gyr_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc29_1_23_17_test6hole_noIa_30Gyr_bridges'):
                rundir='mw_cr_lr_dc29_1_23_17_test6hole_noIa_30Gyr_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e29'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='mw_cr_lr_dc28_1_23_17_test6_noIa_30Gyr_bridges'):
                rundir='mw_cr_lr_dc28_1_23_17_test6_noIa_30Gyr_bridges'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                maindir='oasis'
        if (runtodo=='FIRE_2_0_or_h573_criden1000_noaddm_sggs'):
                rundir='FIRE_2_0_or_h573_criden1000_noaddm_sggs'
                slabel='no CR'
                havecr=0
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                cosmo=1
		maindir='oasis'

        if (runtodo=='FIRE_2_0_h573_CRtest2'):
                rundir='FIRE_2_0_h573_CRtest2'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                cosmo=1
                maindir='/oasis/'

        if (runtodo=='FIRE_2_0_h573_CRtest2_equaltimestep'):
                rundir='FIRE_2_0_h573_CRtest2_equaltimestep'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                cosmo=1
                maindir='/oasis/'

        if (runtodo=='FIRE_2_0_h553_CRtest2'):
                rundir='FIRE_2_0_h553_CRtest2'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                cosmo=1
                maindir='/oasis/'


        if (runtodo=='FIRE_2_0_h553_CRtest2'):
                rundir='FIRE_2_0_h553_CRtest2'
                slabel='CR'
                havecr=6
                runtitle='MW'
                snlabel=r'$f_{mec}=1$'
                dclabel='3e28'
                resolabel='lr'
                Fcal=0.1
                iavesfr = 1.0
                subdir='/output/'
                cosmo=1
                maindir='/oasis/'


	Nsnapstring = str(Nsnap)
        if Nsnap<10:
                Nsnapstring = '00'+str(Nsnap)
        elif Nsnap<100:
                Nsnapstring = '0'+str(Nsnap)
        the_snapdir = '/home/tkc004/'+maindir+'/'+rundir+'/'+subdir
        return {'color':color,'maindir':maindir,'cosmo':cosmo, 'Nsnapstring':Nsnapstring, 'rundir':rundir,'runtitle':runtitle,'slabel':slabel,'snlabel':snlabel,'dclabel':dclabel,'resolabel':resolabel,'the_snapdir':the_snapdir,'havecr':havecr,'Fcal':Fcal,'iavesfr':iavesfr,'timestep':timestep}




def find_enclosed_radius(Srhalf,Slight,Slightneed,ri,si,ncountmax,relerr):
        rlhalf=ri
        slhalf=si
        ncount=0
        while np.absolute((slhalf-Slightneed)/Slightneed)>relerr:
                slhalf = np.sum(Slight[Srhalf<rlhalf])
                rlhalf *= Slightneed/slhalf
                ncount += 1
#                print 'rlhalf, slhalf', rlhalf, slhalf
                if ncount>ncountmax:
                        break
	rhalf = rlhalf
	shalf = slhalf
        return rhalf, shalf

def readhalos(dirname,halostr,hubble=0.702):
        halofile=open(dirname+'/halos/halo_000'+halostr+'.dat','r')
        halofile.readline()
        halofile.readline()
        dars = halofile.readlines()
        halofile.close()
        zlist=[]
        idlist=[]
        mvirlist=[]
        fMhireslist=[]
        mgaslist=[]
        mstarlist=[]
        rvirlist=[]
        haloXl=[]
        haloYl=[]
        haloZl=[]
        for line in dars:
                xsd = line.split()
                zlist.append(float(xsd[0]))
                idlist.append(int(xsd[1]))
                mvirlist.append(float(xsd[4])/hubble)
                haloXl.append(float(xsd[6])/hubble)
                haloYl.append(float(xsd[7])/hubble)
                haloZl.append(float(xsd[8])/hubble)
                fMhireslist.append(float(xsd[38]))
                mgaslist.append(float(xsd[54])/hubble)
                mstarlist.append(float(xsd[74])/hubble)
                rvirlist.append(float(xsd[12])/hubble)
	zlist=np.array(zlist)
	a_scale = 1./(1.+zlist)
	idlist=np.array(idlist)
	mvirlist=np.array(mvirlist)
	fMhireslist=np.array(fMhireslist)
	#print 'a_scale', a_scale
	#print 'haloXl', haloXl
	haloX=np.array(haloXl)*a_scale
	haloY=np.array(haloYl)*a_scale
	haloZ=np.array(haloZl)*a_scale
	#print 'haloX', haloX
	mgaslist=np.array(mgaslist)
	mstarlist=np.array(mstarlist)
	rvirlist=np.array(rvirlist)
	return {'k':1,'mv':mvirlist,'fM':fMhireslist,'haloX':haloX,'haloY':haloY,'haloZ':haloZ,'mg':mgaslist,'ms':mstarlist,'rv':rvirlist,'z':zlist,'id':idlist};	


def reff_ell(H, edges, haloX, haloY, haloZ, nobin, den):
        NominS=[]
        ellX=[]
        ellY=[]
        ellZ=[]
        Spos=[]
        Sposx=[]
        Sposy=[]
        Sposz=[]
	errlist=[]
        SXi=haloX
        SYi=haloY
        SZi=haloZ
        nit=0
        massrat=0
        upno=1000
        upmr=0.45
        dmr=0.55
        nopa=float(den*200)
        NominS=np.append(NominS,nopa)
        nopau=float(den*400)
        nopad=float(den*10)
        inell=0
        while (nit<1000 and (massrat<dmr or massrat>upmr)):
                nit+=1
                if inell==1:
                        nopa=(nopau+nopad)/2.
                NominS=np.append(NominS,nopa)

                print 'nopa', nopa
                print 'len(Sposx)', len(Sposx)
                Spos=[]
                Sposx=[]
                Sposy=[]
                Sposz=[]
                totalno=0
                for i in range(nobin):
                        for j in range(nobin):
                                for k in range(nobin):
                                        if (H[i,j,k]>nopa):
                                                Sposx=np.append(Sposx,edges[0][i])
                                                Sposy=np.append(Sposy,edges[1][j])
                                                Sposz=np.append(Sposz,edges[2][k])
                                                totalno+=H[i,j,k]
                print 'len(Sposx)', len(Sposx)
                if ((len(Sposx)>3 and len(Sposx)<upno) or inell==1):
                        inell=1
                        #print 'fitting ell'
                        points = np.vstack(((Sposx+haloX),(Sposy+haloY),(Sposz+haloZ))).T
                        A, centroid = mvee(points)
                        print 'centroid', centroid
                        SXi=centroid[0]
                        SYi=centroid[1]
                        SZi=centroid[2]
                        U, D, V = la.svd(A)
                        rx, ry, rz = 1./np.sqrt(D)
                        u, v = np.mgrid[0:2*pi:20j, -pi/2:pi/2:10j]

                        def ellipse(u,v):
                            x = rx*cos(u)*cos(v)
                            y = ry*sin(u)*cos(v)
                            z = rz*sin(v)
                            return x,y,z

                        edgespoints=[]
                        E = np.dstack(ellipse(u,v))
                        E = np.dot(E,V) + centroid
                        x, y, z = np.rollaxis(E, axis = -1)
                        err=0
                        errlist=np.append(errlist,0)
                        inV=la.inv(V)
                        for i in range(1,len(edges[0])):
                                for j in range(1,len(edges[1])):
                                        for k in range(1, len(edges[2])):
                                                edgespoints=np.append(edgespoints,(((edges[0][i]+edges[0][i-1])/2-centroid[0]+haloX),((edges[1][j]+edges[1][j-1])/2-centroid[1]+haloY),((edges[2][k]+edges[2][k-1])/2-centroid[2]+haloZ)))
                        edgespoints=np.matrix(edgespoints.reshape((len(edgespoints)/3,3)))
                        rotback=np.dot(edgespoints,inV).T
                        sumHcrit=0
                        for i in range(0,len(edges[0])-1):
                                for j in range(0,len(edges[1])-1):
                                        for k in range(0, len(edges[2])-1):
                                                n=i*(len(edges[2])-1)*(len(edges[2])-1)+j*(len(edges[2])-1)+k
                                                if (np.power(rotback[0,n]/rx,2)+np.power(rotback[1,n]/ry,2)+np.power(rotback[2,n]/rz,2)<1):
                                                        sumHcrit=sumHcrit+H[i,j,k]
                        sumH=np.sum(H)
                        massrat=sumHcrit/sumH
                        print 'sum in ell/all', massrat
                        del x, y, z, E, points, A, centroid,U, D, V,u, v, inV, rotback
                        massratm=massrat
                        nopam=nopa
                        if massratm>upmr:
                                nopad=nopa*1.2
                        if massratm<dmr:
                                nopau=nopa*0.8
                        if (nit>3):
#                               print 'NominS', NominS[nit-1], NominS[nit-3]
                                if (np.abs(NominS[nit-1]-NominS[nit-3])<0.0001*NominS[nit-1]):
                                        err=5
                                        errlist=np.append(errlist,5)
                                        break
                        if (sumHcrit<1e-8):
                                err=6
                                errlist=np.append(errlist,6)
                                break
                elif (inell==0):
                        if(len(Sposx)<3):
                                nopa=nopa*0.9-1
                                nopau=nopa
                        if(len(Sposx)>upno):
                                nopa=nopa*1.1+1
                                nopad=nopa
	return SXi, SYi, SZi, rx, ry, rz, err, massrat, Sposx


def cosmichalo(runtodo):
	multifile='n'
	subdir='hdf5/'
	beginno=400
	finalno=441
	halocolor='k'
	labelname='not set'
	xmax_of_box=40.0
	halostr='00'
	firever=1
	usepep=0
	maindir='scratch'
	highres=0
	fileno=4
	if (runtodo=='m09'):
		rundir='m09_hr_Dec16_2013/'
		halostr='00'
	if (runtodo=='m10'):
		rundir='m10_hr_Dec9_2013/'
		halostr='00'
		subdir='/'
	if (runtodo=='m11'):
		rundir='m11_hhr_Jan9_2013/'
		halostr='01'
		halocolor='m'
		labelname='m11'
	if (runtodo=='m12v'):
		rundir='m12v_mr_Dec5_2013_3/'
		halo_to_do=[0]
	if (runtodo=='B1'):
		rundir='B1_hr_Dec5_2013_11/'
		halo_to_do=[0]
	if (runtodo=='m12qq'):
		rundir='m12qq_hr_Dec16_2013/'
		halo_to_do=[0]
	if (runtodo=='383'):
		rundir='hydro_l8l_z3_383/'
		halostr='00'
		halocolor='y'
		labelname='m11h383'
	if (runtodo=='476'):
		rundir='hydro_l10l_z3_476/'
		halostr='00'
		halocolor='c'
		labelname='m10h476'
	if (runtodo=='553'):
		rundir='hydro_l10l_z3_553/'
		halostr='00'
		halocolor='r'
		labelname='m10h553'
	if (runtodo=='573'):
		rundir='hydro_l10l_z3_573/'
		halostr='00'
		halocolor='g'
		labelname='m10h573'
        if (runtodo=='f383'):
                rundir='FIRE_2_0_or_h383_criden1000_noaddm_sggs/'
                halostr='00'
                subdir='output'
                beginno=100
                finalno=600
		xmax_of_box=40.0
		halocolor='y'
		labelname='m11c'
		firever=2
        if (runtodo=='f383_hv'):
                rundir='FIRE_2_0_or_h383_hv/'
                halostr='00'
                subdir='output'
                beginno=100
                finalno=600
                xmax_of_box=40.0
                halocolor='y'
                labelname='m11c_hv'
                firever=2
		maindir='oasis'
		usepep=0
		highres=1
        if (runtodo=='fm11'):
                rundir='FIRE_2_0_m11/'
                halostr='01'
                subdir='output'
                beginno=100
                finalno=600
                xmax_of_box=40.0
		multifile='y'
		halocolor='m'
		labelname='m11q'
		firever=2
	if (runtodo=='f573'):
		rundir='FIRE_2_0_or_h573_criden1000_noaddm_sggs/'
		halostr='00'
		subdir='output'
		beginno=100
		finalno=600
                xmax_of_box=40.0
		halocolor='g'
		labelname='m10z'
		firever=2
        if (runtodo=='f573_hv'):
                rundir='FIRE_2_0_or_h573_hv/'
                halostr='00'
                subdir='output'
                beginno=100
                finalno=600
                xmax_of_box=40.0
                halocolor='g'
                labelname='m10z_hv'
                firever=2
		maindir='oasis'
		usepep=1
		highres=1
		multifile='y'
        if (runtodo=='f476'):
                rundir='FIRE_2_0_or_h476_noaddm/'
                halostr='00'
                subdir='output'
                beginno=100
                finalno=600
                xmax_of_box=40.0
		halocolor='c'
		labelname='m11b'
                firever=2
        if (runtodo=='f553'):
                rundir='FIRE_2_0_or_h553_criden1000_noaddm_sggs/'
                halostr='00'
                subdir='output'
                xmax_of_box=40.0
                beginno=100
                finalno=600
		halocolor='r'
		labelname='m11a'
                firever=2
	if (runtodo=='1146'):
		rundir='hydro_l10l_z3_1146_ssl/'
		halostr='01'
		halocolor='b'
		labelname='m10h1146'
	if (runtodo=='f1146'):
		rundir='FIRE_2_0_or_h1146_criden1000_noaddm_sggs/'
		halostr='03'
		beginno=100
		finalno=600
		subdir='output'
                xmax_of_box=40.0
		halocolor='b'
		labelname='m10f1146'
                firever=2
		maindir='oasis'
        if (runtodo=='f1146_hv'):
                rundir='FIRE_2_0_or_h1146_hv/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='m10f1146'
                firever=2
		usepep=1
		maindir='oasis'
		highres=1
		multifile='y'
        if (runtodo=='f1297'):
                rundir='FIRE_2_0_or_h1297_criden1000_noaddm_sggs/'
                halostr='03'
                beginno=100
                finalno=600
                xmax_of_box=40.0
		labelname='m10f1297'
                subdir='output'
                firever=2
	if (runtodo=='1297'):
		rundir='hydro_l10l_z3_1297/'
		halostr='02'
        if (runtodo=='df61'):
                rundir='FIRE_2_0_or_h61_dmo/'
                halostr='00'
                beginno=100
                finalno=600
                xmax_of_box=60.0
                labelname='dm11f61'
                subdir='output'
                firever=2
		multifile='y'
        if (runtodo=='f61'):
                rundir='FIRE_2_0_or_h61/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=60.0
                halocolor='b'
                labelname='m11f'
                firever=2
                maindir='oasis'
		multifile='y'
        if (runtodo=='f46'):
                rundir='FIRE_2_0_or_h46/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=60.0
                halocolor='y'
                labelname='m11g'
                firever=2
                maindir='oasis'
                multifile='y'
        if (runtodo=='fm11e'):
                rundir='m11e_res7000/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='m11e'
                firever=2
                maindir='oasis'
                multifile='y'
        if (runtodo=='fm09'):
                rundir='m09/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm09'
                firever=2
                maindir='oasis/extra'
                multifile='y'
		fileno=8
		usepep=1
        if (runtodo=='fm10q'):
                rundir='m10q/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm10q'
                firever=2
                maindir='oasis/extra'
                multifile='y'
                fileno=8
		usepep=1
        if (runtodo=='fm10v'):
                rundir='m10v/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm10v'
                firever=2
                maindir='oasis/extra'
                multifile='y'
                fileno=8
		usepep=1
        if (runtodo=='fm10v'):
                rundir='m10v/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm10v'
                firever=2
                maindir='oasis/extra'
                multifile='y'
                fileno=8
                usepep=1

        if (runtodo=='fm11q'):
                rundir='m11q/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm11q'
                firever=2
                maindir='oasis/extra'
                multifile='y'
                fileno=4
                usepep=1

        if (runtodo=='fm11v'):
                rundir='m11v/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm11v'
                firever=2
                maindir='oasis/extra'
                usepep=1

        if (runtodo=='fm12f'):
                rundir='m12f_ref13/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm12f_ref13'
                firever=2
		multifile='y'
		fileno=4
                maindir='oasis/extra'
                usepep=1

        if (runtodo=='fm12i'):
                rundir='m12i_ref13/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm12i_ref13'
                firever=2
                multifile='y'
                fileno=4
                maindir='oasis/extra'
                usepep=1

        if (runtodo=='fm12m'):
                rundir='m12m_ref13/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm12m_ref13'
                firever=2
                multifile='y'
                fileno=4
                maindir='oasis/extra'
                usepep=1

        if (runtodo=='fm12b'):
                rundir='m12b_ref12/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm12b_ref12'
                firever=2
                maindir='oasis/extra'
                usepep=1


        if (runtodo=='fm12c'):
                rundir='m12c_ref12/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm12c_ref12'
                firever=2
                maindir='oasis/extra'
                usepep=1

        if (runtodo=='fm12q'):
                rundir='m12q_ref12/'
                halostr='00'
                beginno=100
                finalno=600
                subdir='output'
                xmax_of_box=40.0
                halocolor='b'
                labelname='fm12q_ref12'
                firever=2
                maindir='oasis/extra'
                usepep=1


	return {'fileno':fileno,'rundir':rundir,'subdir':subdir,'halostr':halostr,'beginno':beginno,'finalno':finalno, 'multifile':multifile, 'halocolor':halocolor, 'labelname':labelname, 'xmax_of_box':xmax_of_box, 'firever':firever, 'usepep':usepep, 'maindir':maindir, 'highres':highres}

def muprofile(Slight, Sr, murl,bandneeded,UNITS_CGS=1,UNITS_SOLAR_BAND=0):
	Slight=np.array(Slight)
	Sr=np.array(Sr)
	murl=np.array(murl)
	Slin =[]
	for i in range(len(murl)):
		Slin.append(np.sum(Slight[Sr<murl[i]]))
	Slin=np.array(Slin)
	#print 'Slin', Slin
	Slbet = Slin[1:]-Slin[:-1]
	Areabet = np.pi*(np.square(murl[1:])-np.square(murl[:-1]))
	vmag = luminosity_to_magnitude(Slbet, \
        UNITS_CGS=UNITS_CGS, UNITS_SOLAR_BAND=UNITS_SOLAR_BAND,\
        BAND_NUMBER=bandneeded, \
        VEGA=0, AB=1 , \
        MAGNITUDE_TO_LUMINOSITY=0 )
        mul = np.log10(Areabet*1.425*1.425)*2.5+vmag+34.97
        magl = luminosity_to_magnitude(Slin, \
        UNITS_CGS=UNITS_CGS, UNITS_SOLAR_BAND=UNITS_SOLAR_BAND,\
        BAND_NUMBER=bandneeded, \
        VEGA=0, AB=1 , \
        MAGNITUDE_TO_LUMINOSITY=0 )
	return mul, magl



def reff_ell2d(H, edges, haloX, haloY, nobin, den):
        NominS=[]
        ellX=[]
        ellY=[]
        Spos=[]
        Sposx=[]
        Sposy=[]
        errlist=[]
        SXi=haloX
        SYi=haloY
        nit=0
        massrat=0
        upno=1000
        upmr=0.3
        dmr=0.7
        nopa=float(den)
        NominS=np.append(NominS,nopa)
        nopau=float(den*2)
        nopad=float(den*0.2)
        inell=0
	rx=0.
	ry=0.
	SXi=0.
	SYi=0.
        while (nit<100 and (massrat<dmr or massrat>upmr)):
                nit+=1
                if inell==1:
                        nopa=(nopau+nopad)/2.
                NominS=np.append(NominS,nopa)

                print 'nopa', nopa
                print 'len(Sposx)', len(Sposx)
                Spos=[]
                Sposx=[]
                Sposy=[]
                totalno=0
                for i in range(nobin):
                        for j in range(nobin):
				if (H[i,j]>nopa):
					Sposx=np.append(Sposx,edges[0,i])
					Sposy=np.append(Sposy,edges[1,j])
					totalno+=H[i,j]
                print 'len(Sposx)', len(Sposx)
 		print 'totalno', totalno
                if ((len(Sposx)>3 and len(Sposx)<upno) or inell==1):
                        inell=1
		 	#print 'Sposx', Sposx
                        #print 'fitting ell'
                        points = np.vstack(((Sposx+haloX),(Sposy+haloY))).T
                        A, centroid = mvee(points)
                        print 'centroid', centroid
                        SXi=centroid[0]
                        SYi=centroid[1]
                        U, D, V = la.svd(A)
                        rx, ry = 1./np.sqrt(D)
                        u = np.mgrid[0:2*pi:20j]

                        def ellipse2d(u):
                            x = rx*cos(u)
                            y = ry*sin(u)
                            return x,y

                        edgespoints=[]
                        E = np.dstack(ellipse2d(u))
                        E = np.dot(E,V) + centroid
                        x, y= np.rollaxis(E, axis = -1)
                        err=0
                        errlist=np.append(errlist,0)
                        inV=la.inv(V)
                        for i in range(1,nobin):
                                for j in range(1,nobin):
					edgespoints=np.append(edgespoints,(((edges[0,i]+edges[0,i-1])/2-centroid[0]+haloX),((edges[1,j]+edges[1,j-1])/2-centroid[1]+haloY)))
                        edgespoints=np.matrix(edgespoints.reshape((len(edgespoints)/2,2)))
                        rotback=np.dot(edgespoints,inV).T
                        sumHcrit=0
                        for i in range(0,len(edges[0])-1):
                                for j in range(0,len(edges[1])-1):
					n=i*(len(edges[1])-1)+j
					if (np.power(rotback[0,n]/rx,2)+np.power(rotback[1,n]/ry,2)<1):
						sumHcrit=sumHcrit+H[i,j]
                        sumH=np.sum(H)
                        massrat=sumHcrit/sumH
                 #       print 'sum in ell/all', massrat
                        del x, y, E, points, A, centroid,U, D, V,u, inV, rotback
                        massratm=massrat
                        nopam=nopa
                        if massratm>upmr:
                                nopad=nopa*1.1
                        if massratm<dmr:
                                nopau=nopa*0.9
                        if (nit>3):
                #                print 'NominS', NominS[nit-1], NominS[nit-3]
                                if (np.abs(NominS[nit-1]-NominS[nit-3])<0.0001*NominS[nit-1]):
                                        err=5
                                        errlist=np.append(errlist,5)
                                        break
                        if (sumHcrit<1e-8):
                                err=6
                                errlist=np.append(errlist,6)
                                break
                elif (inell==0):
                        if(len(Sposx)<3):
                                nopa=nopa*0.97-1
                                nopau=nopa
                        if(len(Sposx)>upno):
                                nopa=nopa*1.03+1
                                nopad=nopa
        return SXi, SYi, rx, ry,  err, massrat, Sposx

def readtime(firever=2):
	file=open('/home/tkc004/samsonprogram/data/snapshot_times.txt','r')
	file.readline()
	file.readline()
	file.readline()
	dars = file.readlines()
	file.close()
	snap2list=[]
	a2list=[]
	time2list=[]
	red2list=[]
	for line in dars:
		xsd = line.split()
		snap2list.append(int(xsd[0]))
		a2list.append(float(xsd[1]))
		red2list.append(float(xsd[2]))
		time2list.append(float(xsd[3]))
	snap2list=np.array(snap2list)
	time2list=np.array(time2list)
	if firever==1:
		file=open('/home/tkc004/samsonprogram/data/output_times.txt','r')
		dars = file.readlines()
		file.close()
                snaplist=[]
                alist=[]
                timelist=[]
                redlist=[]
		ncount=0
                for line in dars:
                        xsd = line.split()
                        alist.append(float(xsd[0]))
			snaplist.append(ncount)
			ncount+=1
		alist=np.array(alist)
		snaplist=np.array(snaplist)
		timelist=np.array(np.interp(alist,a2list,time2list))
		return snaplist, timelist
	if firever==2:
		return snap2list, time2list

