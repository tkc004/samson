import matplotlib as mpl
mpl.use('Agg')
from readsnap_cr import readsnapcr
import Sasha_functions as SF
import graphics_library as GL
import numpy as np
import matplotlib.pyplot as plt
from samson_functions import *
from matplotlib import rcParams
from pylab import *
from Sasha_functions import *
#rcParams['figure.figsize'] = 5, 5
rcParams['figure.figsize'] = 10, 5
rcParams['font.size']=12
rcParams['font.family']='serif'
#rcParams.update({'figure.autolayout': True})
import matplotlib.patches as patches
rcParams['axes.linewidth'] = 2
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['axes.unicode_minus']=False
colortable = [ 'b', 'g', 'r']
dirneed=['mw_cr_lr_dc29_4_19_M1']
#dirneed=['mw_cr_lr_dc28_3_31_M1']
#dirneed=['mw_cr_lr_dc28_3_31_M1_equaltimestep']
#dirneed=['mw_cr_lr_dc28_3_31_M1_equaltimestep_b4necheck']
#dirneed=['mw_cr_lr_dc29_1_23_17_test6','mw_cr_lr_dc29_1_23_17_test6chole_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test16c_bridges','mw_cr_lr_dc28_1_23_17_test16chole_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6hole_noIa_30Gyr_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test16c_bridges']
#dirneed=['FIRE_2_0_h573_CRtest2_equaltimestep']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test6hole_noIa_30Gyr_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test6hole_noIa_30Gyr_bridges','mw_cr_lr_dc29_1_23_17_test6','mw_cr_lr_dc29_1_23_17_test6hole_noIa_30Gyr_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_testhole','mw_cr_lr_dc28_1_23_17_test6hole_noIa_30Gyr_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_testhole','mw_cr_lr_dc29_1_23_17_test6','mw_cr_lr_dc29_1_23_17_test6chole_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_testhole','mw_cr_lr_dc28_1_23_17_test6_noIa_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6_noIa_bridges','mw_cr_lr_dc28_1_23_17_test6_noIa_30Gyr_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test6_noIa_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test16c_bridges','mw_cr_lr_dc28_1_23_17_test16chole_bridges','mw_cr_lr_dc28_1_23_17_test16cout_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test16chole_bridges']
#dirneed=['mw_cr_lr_dc29_1_23_17_test6']
#dirneed=['mw_cr_lr_dc28_1_23_17_test16c_bridges']
#dirneed=['FIRE_2_0_or_h573_criden1000_noaddm_sggs']
#dirneed=['mw_cr_lr_dc28_1_23_17_testhole']
#dirneed=['FIRE_2_0_or_h573_criden1000_noaddm_sggs']
#dirneed=['FIRE_2_0_h573_CRtest2']
#dirneed=['mw_cr_llr_dc29_1_13_17_test']
#dirneed=['mw_cr_llr_dc28_1_23_17_test16','mw_cr_llr_dc28_1_23_17_testcooling']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6']
#dirneed=['mw_cr_lr_dc28_1_23_17_test16c_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test16c_bridges']
#dirneed=['mw_cr_lr_nodiff_1_23_17_test16c_bridges','mw_cr_lr_dc27_1_23_17_test16c_bridges','mw_cr_lr_dc28_1_23_17_test16c_bridges']
#dirneed=['mw_cr_llr_dc28_1_23_17_test16','mw_cr_lr_dc28_1_23_17_test16c_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc29_1_23_17_test6']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test6','mw_cr_mr_dc28_1_23_17_test6_bridges']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6']
#dirneed=['mw_lr_1_23_17']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_llr_dc28_1_23_17_testhalo']
#dirneed=['mw_cr_lr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_testhole']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_lr_dc28_1_23_17_test6']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6pwu1']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_llr_dc28_1_23_17_testhole','mw_cr_llr_dc28_1_23_17_test16hole']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_llr_dc28_1_23_17_testhole']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_llr_dc28_1_23_17_testout']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_llr_dc28_1_23_17_testhole']
#dirneed=['mw_cr_llr_dc28_1_23_17_test6','mw_cr_llr_dc28_1_23_17_test16']
#dirneed=['mw_cr_llr_dc28_1_23_17_testlargeout','mw_cr_llr_dc28_1_23_17_testout','mw_cr_llr_dc28_1_23_17_testhole','mw_cr_llr_dc28_1_23_17_test6']
#dirneed=['mw_cr_llr_dc28_1_23_17_testlargeout','mw_cr_llr_dc28_1_23_17_testhole','mw_cr_llr_dc28_1_23_17_testout']
#dirneed=['mw_cr_llr_dc28_1_23_17_test8','mw_cr_llr_dc28_1_23_17_test9','mw_cr_llr_dc28_1_23_17_testhole','mw_cr_llr_dc28_1_23_17_testout']
#dirneed=['mw_cr_llr_dc28_1_23_17_test','mw_cr_llr_dc28_1_23_17_test6']
#dirneed=['mw_cr_llr_dc28_1_13_17_test','mw_cr_llr_dc28_1_13_17_test3']
#dirneed=['mw_cr_llr_dc28_1_13_17_test','mw_cr_llr_dc28_1_13_17_test3']
#dirneed=['mw_cr_llr_dc28_1_13_17_test','mw_cr_llr_dc28_1_23_17_test']
#dirneed=['mw_cr_llr_nodiff_1_13_17_test','mw_cr_llr_dc28_1_13_17_test','mw_cr_llr_dc28_5_1_13_17_test','mw_cr_llr_dc29_1_13_17_test']
#dirneed=['mw_cr_llr_dc28_1_13_17_test','mw_cr_llr_dc28_1_23_17_test']
#dirneed=['mw_cr_llr_dc29_1_13_17_test1','mw_cr_llr_dc29_1_13_17_test2','mw_cr_llr_dc29_1_13_17_test3','mw_cr_llr_dc29_1_13_17_test4']
#dirneed=['mw_cr_llr_dc28_1_13_17_test1']
#dirneed=['mw_cr_llr_dc28_1_13_17_test', 'mw_cr_llr_dc29_1_13_17_test']
#fmeat='equaltimestep'
fmeat='M1'
#fmeat='hole16c'
#fmeat='hole29'
#fmeat='holenoIa'
#fmeat='573'
#fmeat='sf_model_tests_mhd_cr_d10_hydro'
#fmeat='lrdc28_noIa'
#fmeat='lrdc28hole_noIa'
#fmeat='lrdc2816c'
#fmeat='hole'
#fmeat='lrdc28holeout'
#fmeat='test6_16'
#fmeat='test16c_dc28_lr_llr'
#fmeat='13_23_mw'
#fmeat= 'mw_cr_llr_dc28_test_Jan'
startno=0
#Nsnap=371
Nsnap=110
#Nsnap=0
#Nsnap=600
snapsep=10
#wanted = 'gasdenmidplane'
#wanted = 'sfrv'
#wanted = 'sfrrad'
#wanted = 'sfrarad'
#wanted='sncretime'
#wanted='crdenplanes'
#wanted='crdenv'
#wanted='crdenmidplane'
#wanted='gammadensph'
#wanted='credensph'
#wanted='decayratiosph'
#wanted='decaytimesph'
#wanted='gammasph'
#wanted='Gmencsph'
#wanted='gasdensph'
#wanted='gasden'
#wanted='gmr'
#wanted='crer'
#wanted='crez'
#wanted='gammar'
#wanted='gammaz'
#wanted='cresph'
#wanted='gmz'
#wanted='gasdenz'
#wanted='nismgamma'
#wanted='nismcumgamma'
#wanted='nismcumgm'
#wanted='gammadecaytime'
#wanted='gammacumdecaytime'
#wanted='credecaytime'
#wanted='crecumdecaytime'
#wanted='nismcumcre'
#wanted = 'dirgammasfr'
#wanted = 'crasnap'
wanted='cratime'
#wanted='gasdenv'
#wanted='outflowwrite'
#wanted='crdrad'
#wanted='cramap'
#wanted='cramapv'
#wanted='outflowwrite'
#wanted = 'gassurden'
#wanted='outflow'
#wanted='gasfrac'
#wanted='printparmass'
#wanted='enchange'
#wanted='dirage'
#wanted='dirsm'
#wanted='dirsfr'
#wanted='dirgammasfr'
#wanted='cosmic_raywrite'
#wanted='crdensol' #cosmic ray energy density at solar circle (observation ~ 1 eV/cm^3)
#wanted='smchange'
#wanted='cumgamma'
#wanted='smass'
#wanted='crdtime'
#wanted='crchange'
#wanted='crtime'
#wanted='crcooling'
#wanted='lgratio'
#wanted='dg_sfr'
#wanted='gamma_partit'
#wanted='dlgratio'
#wanted='gsmratio'
#wanted='dirgammasfr'
useM1=1
the_prefix='snapshot'
the_suffix='.hdf5'
withinr=20.0
nogrid=20
maxlength=0.5
med = -0.1 
wholeboxgas=1
diskgas=1
betapi = 0.7 ##should be between 0.5-0.9 from Lacki 2011 #fraction of pi has enough energy (>1GeV)
sec_in_yr = 3.2e7
nopi_per_gamma = 3.0
solar_mass_in_g = 2e33
km_in_cm = 1e5
kpc_in_cm = 3.086e21
erg_in_eV = 6.242e11
cspeed_in_cm_s = 3e10
proton_mass_in_g = 1.67e-24
Kroupa_Lsf=6.2e-4
Salpeter_Lsf=3.8e-4
pidecay_fac = 2.0e5*250.0*sec_in_yr #over neff in cm_3 for pi decay time in s

if wanted=='printparmass':
        for runtodo in dirneed:
                for i in [0]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        havecr=0
			DM = readsnap(the_snapdir, Nsnapstring, 1, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			B = readsnap(the_snapdir, Nsnapstring, 3, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			D = readsnap(the_snapdir, Nsnapstring, 3, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			#print 'DMmass', DM['m']*1e10
			#print 'Bmass', B['m']*1e10
			#print 'Dmass', D['m']*1e10
			#print 'Gmass', G['m']*1e10
			print 'dmmass', np.average(DM['m'])	
			print 'bmass', np.average(B['m'])
			print 'dmass', np.average(D['m'])
			print 'gasmass', np.average(G['m']) 

if wanted=='gasfrac':
        for runtodo in dirneed:
		gbflb = []
		gbfl = []
		Nsnapl = []
                for i in range(0,Nsnap,snapsep):
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
			havecr=0
			G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			DM = readsnap(the_snapdir, Nsnapstring, 1, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			disk = readsnap(the_snapdir, Nsnapstring, 2, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			bulge = readsnap(the_snapdir, Nsnapstring, 3, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			Gp = G['p']
			Grho = G['rho']
			Gu = G['u']
			Gm = G['m']
			diskm = disk['m']
			bulgem = bulge['m']
			try:
				Sm = S['m']
			except KeyError:
				Sm=[]
			DMm = DM['m']
			Gz = Gp[:,2]
			Gx = Gp[:,0]
			Gy = Gp[:,1]
			cutxy = Gx*Gx+Gy*Gy < withinr*withinr
			cutz = np.absolute(Gz) < maxlength
			cut = cutxy*cutz
			gasbarf = np.sum(Gm[cut])/(np.sum(Gm[cut])+np.sum(Sm)+np.sum(diskm)+np.sum(bulgem))
                        gasbarfbox = np.sum(Gm)/(np.sum(Gm)+np.sum(Sm)+np.sum(diskm)+np.sum(bulgem))
			Nsnapl = np.append(Nsnapl, i)
			gbfl = np.append(gbfl,gasbarf)
			gbflb = np.append(gbflb, gasbarfbox)
			#cut = Gx*Gx+Gy*Gy < withinr*withinr
		#plt.plot(Nsnapl*0.001, gbfl)
		if wholeboxgas==1:
			plt.plot(Nsnapl, gbflb, label=runtodo+'_box')
		if diskgas==1:
			plt.plot(Nsnapl, gbfl, label=runtodo+'_disk')
	plt.xlabel('Myr')
	plt.ylabel('Gas fraction')
	plt.legend(loc='best')
	if wholeboxgas==1:
		plt.savefig('gasfraction_'+fmeat+'_box.pdf')
	else:
		plt.savefig('gasfraction_'+fmeat+'.pdf')
	plt.clf()	


if wanted=='outflow':
	for signal in [20,8,16,5]:
		for runtodo in dirneed:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
			spmname='outflow_'+runtodo+'.txt'
			spmfile=open(spmname,"r")
			spmfile.readline()
			dars = spmfile.readlines()

			snapno=[]
			outflow=[]
			outflow16=[]
			outflow8=[]
			outflow0_5=[]
			for line in dars:
				xsd = line.split()
				snapno = np.append(snapno, int(xsd[0]))
				outflow = np.append(outflow, float(xsd[1]))
				outflow16 = np.append(outflow16, float(xsd[2]))
				outflow8 = np.append(outflow8, float(xsd[3]))
				outflow0_5 = np.append(outflow0_5, float(xsd[4]))		
			spmfile.close()

			if havecr==0:
				needlabel = slabel
			else:
				needlabel = r' $\kappa_{\rm di}=$'+dclabel

			if signal==20:
				plt.plot(snapno*1e-3, outflow, label=needlabel)
				plt.title('0-20kpc')
			if signal==8:
				plt.plot(snapno*1e-3, outflow8, label=needlabel)
				plt.title('8-12kpc')
			if signal==16:
				plt.plot(snapno*1e-3, outflow16, label=needlabel)
				plt.title('16-24kpc')
			if signal==5:
				plt.plot(snapno*1e-3, outflow0_5, label=needlabel)
				plt.title('>0.5kpc and v > 1000 km/s')
			plt.xlabel('Gyr')
			plt.ylabel('outflow')
			plt.xlim(xmax=Nsnap*1e-3)
		plt.yscale('log')
		plt.legend(loc='best')
		if signal==20:
			plt.savefig('outflow_'+runtodo+'.pdf')
		if signal==8:
			plt.savefig('outflow8_'+runtodo+'.pdf')
		if signal==16:
			plt.savefig('outflow16_'+runtodo+'.pdf')
		if signal==5:
			plt.savefig('outflow0_5_'+runtodo+'.pdf')
		plt.clf()

if wanted=='sfr':
	for runtodo in dirneed:
		try:
			rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
			S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
			Sage = S['age']
		except KeyError:
			Nsnap-=1
			rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
                        S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
                        Sage = S['age']
		#aveSmass = np.average(S['m'])
		hist, bin_edges = np.histogram(Sage, bins=10, weights=S['m'])
		print 'bin_edges', bin_edges
		print 'hist', hist
		if havecr==0:
			needlabel = slabel
		else:
			needlabel = r' $\kappa_{\rm di}=$'+dclabel
		plt.plot(bin_edges[1:]*0.98, hist/(bin_edges[1]-bin_edges[0])/0.98*10, label=needlabel)
		plt.xlabel('Gyr')
		plt.ylabel(r'SFR ($M_\odot$/yr)')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.title(runtitle)
	plt.savefig('starage_'+fmeat+'.pdf')
	plt.clf()

if wanted=='pregrad':
	for runtodo in dirneed:
		rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr , Fcal, iavesfr =outdirname(runtodo, Nsnap)
		G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
		Gp = G['p']
		Grho = G['rho']
		Gu = G['u']
		Gm = G['m']
		if (havecr==1):
			Gcregy=G['cregy']
		Gz = Gp[:,2]
		Gx = Gp[:,0]
		Gy = Gp[:,1]
		cut = Gx*Gx+Gy*Gy < withinr*withinr
		if (havecr==1):
			Gcregyc=Gcregy[cut]*2e53
		Gzcut=Gz[cut]
		Grhoc=Grho[cut]*7e-22
		Guc=Gu[cut]*1e10
		Gmc=Gm[cut]*1e10*2e33
		heightlist=[]
		pgradlist=[]
		Gprelist=[]
		gamma=1.5
		preGpre=0.0
		for i in range(nogrid):
			height = i*maxlength/nogrid-maxlength/2.0
			big=Gzcut>height
			small=Gzcut<height+maxlength/nogrid
			within=big*small
			Gpre0=(gamma-1.0)*Guc[within]*Grhoc[within]
			if (havecr==1):
				Gpre=Gpre0+(0.3333)*Gcregyc[within]*Grhoc[within]/Gmc[within]
				#print 'ratio', Gpre0/Gpre
			else:
				Gpre=Gpre0
			heightlist=np.append(heightlist,height)
			Gprelist=np.append(Gprelist,np.average(Gpre))
			#pgradlist=np.append(pgradlist,np.absolute(np.average(Gpre)-preGpre)*nogrid/maxlength/3e21)
			#preGpre=np.average(Gpre)
		#print 'Gprelist', len(Gprelist)
		#print 'heightlist', len(heightlist)
		#plt.plot(Gprelist,heightlist,label=slabel)
                if havecr==0:
                        needlabel = slabel
                else:
                        needlabel = r' $\kappa_{\rm di}=$'+dclabel
		plt.plot(np.absolute((Gprelist[:-1]-Gprelist[1:])/(heightlist[:-1]-heightlist[1:]))/3.09e21,(heightlist[1:]+heightlist[:-1])*0.5,label=needlabel)
	plt.legend(loc='best',fontsize=14)
	plt.title(runtitle)
	#plt.title('pressure gradient in a cylinder of radius '+str(withinr)+'kpc centered on the '+runtitle+' disk')
	#plt.xlim([-300,300])
	plt.ylim([-maxlength/2, maxlength/2])
	plt.ylabel('z (kpc)')
	plt.xlabel(r'${\rm dP/dz \; (g/cm^2/s^2)}$')
	plt.xscale('log')
	plt.savefig('pgrad_disk_'+runtodo+'_'+str(withinr)+'kpc.pdf')

if wanted=='pre':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
                G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                Gp = G['p']
                Grho = G['rho']
                Gu = G['u']
                Gm = G['m']
                if (havecr==1):
                        Gcregy=G['cregy']
                Gz = Gp[:,2]
                Gx = Gp[:,0]
                Gy = Gp[:,1]
                cut = Gx*Gx+Gy*Gy < withinr*withinr
                if (havecr==1):
                        Gcregyc=Gcregy[cut]*2e53
                Gzcut=Gz[cut]
                Grhoc=Grho[cut]*7e-22
                Guc=Gu[cut]*1e10
                Gmc=Gm[cut]*1e10*2e33
                heightlist=[]
                pgradlist=[]
                Gprelist=[]
                gamma=1.5
                preGpre=0.0
                for i in range(nogrid):
                        height = i*maxlength/nogrid-maxlength/2.0
                        big=Gzcut>height
                        small=Gzcut<height+maxlength/nogrid
                        within=big*small
                        Gpre0=(gamma-1.0)*Guc[within]*Grhoc[within]
                        if (havecr==1):
                                Gpre=Gpre0+(0.3333)*Gcregyc[within]*Grhoc[within]/Gmc[within]
                        else:
                                Gpre=Gpre0
                        heightlist=np.append(heightlist,height)
                        Gprelist=np.append(Gprelist,np.average(Gpre))
                        #pgradlist=np.append(pgradlist,np.absolute(np.average(Gpre)-preGpre)*nogrid/maxlength/3e21)
                        #preGpre=np.average(Gpre)
                #print 'Gprelist', len(Gprelist)
                #print 'heightlist', len(heightlist)
                #plt.plot(Gprelist,heightlist,label=slabel)
                if havecr==0:
                        needlabel = slabel
                else:
                        needlabel = r' $\kappa_{\rm di}=$'+dclabel
                plt.plot(Gprelist,heightlist,label=needlabel)
        plt.legend(loc='best')
        plt.title(runtitle)
        #plt.title('pressure gradient in a cylinder of radius '+str(withinr)+'kpc centered on the '+runtitle+' disk')
        #plt.xlim([-300,300])
        plt.ylim([-maxlength/2, maxlength/2])
        plt.ylabel('z (kpc)')
        plt.xlabel(r'${\rm P \; (g/cm/s^2)}$')
        plt.xscale('log')
        plt.savefig('pre_disk_'+runtodo+'_'+str(withinr)+'kpc.pdf')


if wanted=='zv':
	for runtodo in dirneed:
		rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr =outdirname(runtodo, Nsnap)
		G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix)
		Gp = G['p']
		Gv = G['v']
		Gm = G['m']
		Gvz = Gv[:,2]
		Gz = Gp[:,2]
		Gx = Gp[:,0]
		Gy = Gp[:,1]
		h = 1

		#find particles within 3kpc cylinder in z plane:
		cut = Gx*Gx+Gy*Gy < withinr*withinr
		Gzcut=Gz[cut]
		Gmc=Gm[cut]
		Gvzc=Gvz[cut]
		heightlist=[]
		medvlist=[]

		for i in range(nogrid):
			height = i*maxlength/nogrid-maxlength/2.0
			big=Gzcut>height
			small=Gzcut<height+maxlength/nogrid
			within=big*small
			Gmcwi=Gmc[within]
			Gvzcwi=Gvzc[within]
			#print 'Gvzcwi', Gvzcwi
			heightlist=np.append(heightlist,height)
			medvlist=np.append(medvlist,np.median(Gvzcwi))
                if havecr==0:
                        needlabel = slabel
                else:
                        needlabel = r' $\kappa_{\rm di}=$'+dclabel
		plt.plot(medvlist,heightlist,label=needlabel)
	plt.legend(loc='best')
	#plt.title('Median Vz in a cylinder of radius 10kpc centered on the galactic disk')
	plt.title(runtitle)
	plt.xlim([-400,400])
	plt.ylim([-maxlength/2, maxlength/2])
	plt.ylabel('z (kpc)')
	plt.xlabel('Vz (km/s)')
	plt.savefig('vzdisk_'+runtodo+'_'+str(withinr)+'kpc.pdf')



if wanted=='massloading':
	for signal in [20,8,16, 5]:
		for runtodo in dirneed:
			rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
			spmname='outflow_'+runtodo+'.txt'
			spmfile=open(spmname,"r")
			spmfile.readline()
			dars = spmfile.readlines()

			snapno=[]
			outflow=[]
			outflow16=[]
			outflow8=[]
			outflow0_5=[]
			smlist=[]

			for line in dars:
				xsd = line.split()
				snapno = np.append(snapno, int(xsd[0]))
				outflow = np.append(outflow, float(xsd[1]))
				outflow16 = np.append(outflow16, float(xsd[2]))
				outflow8 = np.append(outflow8, float(xsd[3]))
				outflow0_5 = np.append(outflow0_5, float(xsd[4]))
				smlist = np.append(smlist, float(xsd[5]))
			spmfile.close()
			sfr = (smlist[1:]-smlist[:-1])/(snapno[1:]-snapno[:-1])/0.98*10
			print 'sfr', sfr
			massloading=outflow[:-1]/sfr
			massloading8 = outflow8[:-1]/sfr
			massloading16 = outflow16[:-1]/sfr
			massloading0_5 = outflow0_5[:-1]/sfr
			if havecr==0:
				needlabel = slabel
			else:
				needlabel = r' $\kappa_{\rm di}=$'+dclabel

			if signal==20:
				plt.plot((snapno[:-1])*0.98, massloading, label=needlabel)
				plt.title(runtitle+' 0-20kpc')
			if signal==8:
				plt.plot((snapno[:-1])*0.98, massloading8, label=needlabel)
				plt.title(runtitle+' 8-12kpc')
			if signal==16:
				plt.plot((snapno[:-1])*0.98, massloading16, label=needlabel)
				plt.title(runtitle+' 16-24kpc')
			if signal==5:
				plt.plot((snapno[:-1])*0.98, massloading0_5, label=needlabel)
				plt.title('>0.5kpc and v > 100 km/s')
			plt.xlabel('Gyr')
			plt.ylabel(r'$\eta$', fontsize=24)
		plt.yscale('log')
		plt.legend(loc='best')
		if signal==20:
			plt.savefig('massloading_'+runtodo+'.pdf')
		if signal==8:
			plt.savefig('massloading8_'+runtodo+'.pdf')
		if signal==16:
			plt.savefig('massloading16_'+runtodo+'.pdf')
		if signal==5:
			plt.savefig('massloading0_5_'+runtodo+'.pdf')
		plt.clf()



if wanted=='smchange':
	for runtodo in dirneed:
		info=outdirname(runtodo, Nsnap)
		rundir=info['rundir']
		runtitle=info['runtitle']
		slabel=info['slabel']
		snlabel=info['snlabel']
		dclabel=info['dclabel']
		resolabel=info['resolabel']
		the_snapdir=info['the_snapdir']
		Nsnapstring=info['Nsnapstring']
		havecr=info['havecr']
		Fcal=info['Fcal']
		iavesfr=info['iavesfr']
		timestep=info['timestep']
		spmname='outflow_'+runtodo+'.txt'
		spmfile=open(spmname,"r")
		spmfile.readline()
		dars = spmfile.readlines()

		snapno=[]
		smlist=[]
		if havecr==0:
			needlabel = slabel
		else:
			needlabel = r' $\kappa_{\rm di}=$'+dclabel

		for line in dars:
			xsd = line.split()
			snapno = np.append(snapno, int(xsd[0]))
			smlist = np.append(smlist, float(xsd[5]))
		spmfile.close()
                smlists=smlist[::snapsep]
                snapnos=snapno[::snapsep]
		sfr = (smlists[1:]-smlists[:-1])/(snapnos[1:]-snapnos[:-1])/0.98*1e4
		print 'sfr', sfr
		#plt.plot((snapno[1:]+snapno[:-1])/2.*0.98, sfr, label=needlabel)
                #plt.plot((snapnos[1:]+snapnos[:-1])/2.*0.98, sfr, label='kappa='+dclabel+'; '+'time step= '+timestep)
		#plt.plot((snapnos[1:]+snapnos[:-1])/2.*0.98, sfr, label='kappa='+dclabel)
		plt.plot((snapnos[1:]+snapnos[:-1])/2.*0.98, sfr, label=runtodo)
	plt.legend(loc='best')
	plt.title(runtitle)
	plt.xlim(xmin=float(startno),xmax=float(Nsnap))
	plt.yscale('log')
	plt.ylabel(r'SFR $({\rm M}_{\odot}/{\rm yr})$')
	plt.xlabel('Myr')
	plt.savefig('CRplot/sfr_'+fmeat+'.pdf')
	plt.clf()

if wanted=='smass':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
                spmname='outflow_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()

                snapno=[]
                smlist=[]
                if havecr==0:
                        needlabel = slabel
                else:
                        needlabel = r' $\kappa_{\rm di}=$'+dclabel

                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        smlist = np.append(smlist, float(xsd[5]))
                spmfile.close()
                #plt.plot((snapno[1:]+snapno[:-1])/2.*0.98, sfr, label=needlabel)
                plt.plot(snapno*0.98, smlist*1e10, label=runtodo)
        plt.legend(loc='best')
        plt.title(fmeat)
        plt.xlim(xmax=float(Nsnap))
        plt.yscale('log')
        plt.ylabel(r' $M_{*}({\rm M}_{\odot})$')
        plt.xlabel('Myr')
        plt.savefig('smass_'+fmeat+'.pdf')
        plt.clf()
	

if wanted=='gassurden':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
                spmname='outflow_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()

                snapno=[]
                smlist=[]
		sm1kpc=[]
		gm1kpc=[]
                if havecr==0:
                        needlabel = slabel
                else:
                        needlabel = r' $\kappa_{\rm di}=$'+dclabel

                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        smlist = np.append(smlist, float(xsd[5]))
			sm1kpc = np.append(sm1kpc, float(xsd[6]))
			gm1kpc = np.append(gm1kpc, float(xsd[7]))
                spmfile.close()
                #sfr = (smlist[1:]-smlist[:-1])/(snapno[1:]-snapno[:-1])/0.98*1e4
		sfr1kpc = (sm1kpc[1:]-sm1kpc[:-1])/(snapno[1:]-snapno[:-1])/0.98*1e4
		#plt.plot(np.log10((gm1kpc[1:]+gm1kpc[:-1])/2.*1e10/np.pi/1e6),np.log10(sfr/np.pi), ls='none', marker='o')
		plt.plot(np.log10((gm1kpc[1:]+gm1kpc[:-1])/2.*1e10/np.pi/1e6),np.log10(sfr1kpc/np.pi), ls='none', marker='o', label=runtodo)
                #plt.plot((snapno[1:]+snapno[:-1])/2.*0.98, sfr, label=needlabel)
		#plt.plot((snapno[1:]+snapno[:-1])/2.*0.98, sfr1kpc, label='< 1kpc')
        plt.legend(loc='best')
        plt.title(fmeat)
        plt.ylabel(r'$\log (\sum_{\rm SFR} [{\rm M}_{\rm sun}/{\rm yr}/{\rm kpc^2}])$')
        plt.xlabel(r'$\log (\sum_{\rm gas} [{\rm M}_{\rm sun}/{\rm pc^2}])$')
        plt.savefig('KSlaw_1kpc_'+fmeat+'.pdf')
        plt.clf()


if wanted=='outflowwrite':
	for runtodo in dirneed:
		snaplist=[]
		outlist=[]
		outlist16=[]
		outlist8=[]
		outlist0_5=[]
		outlistpb=[]
		mslist=[]
		gm1kpc=[]
		sm1kpc=[]
        	for i in range(0,Nsnap,snapsep):
			veccut=100
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
			print 'the_snapdir', the_snapdir
			print 'Nsnapstring', Nsnapstring
			G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix)
			Gp = G['p']
			Gv = G['v']
			Gm = G['m']
			Gvx = Gv[:,0]
			Gvy = Gv[:,1]
			Gvz = Gv[:,2]
			Gx = Gp[:,0]
			Gy = Gp[:,1]
			Gz = Gp[:,2]
			#find particles within 1kpc
			in1kpc = np.square(Gx)+np.square(Gy)+np.square(Gz)<1
			gasmass1kpc = np.sum(Gm[in1kpc])
			

			#find particles within 0-20, 16-24, 8-12kpc in z direction:
			up = Gz>0
			down = Gz<0
			within = np.absolute(Gz) < 20
			withinu = up*within
			withind = down*within

			up16 = Gz>16
			down16 = Gz<-16
			within16 = np.absolute(Gz) < 24
			withinu16 = up16*within16
			withind16 = down16*within16

			up8 = Gz>8
			down8 = Gz<-8
			within8 = np.absolute(Gz) < 12
			withinu8 = up8*within8
			withind8 = down8*within8
			#find particles above 500pc from the disk
			up0_5 = Gz> 0.5
			down0_5 = Gz <-0.5
			Gviu = Gvz[withinu]
			Gvid = Gvz[withind]
			Gmiu = Gm[withinu]
			Gmid = Gm[withind]
			print 'Gviu', Gviu
			upout = Gviu>veccut
			downout = Gvid<veccut

			Gviu16 = Gvz[withinu16]
			Gvid16 = Gvz[withind16]
			Gmiu16 = Gm[withinu16]
			Gmid16 = Gm[withind16]
			upout16 = Gviu16>veccut
			downout16 = Gvid16<veccut



			Gviu8 = Gvz[withinu8]
			Gvid8 = Gvz[withind8]
			Gmiu8 = Gm[withinu8]
			Gmid8 = Gm[withind8]
			upout8 = Gviu8>veccut
			downout8 = Gvid8<veccut

			Gvu0_5 = Gvz[up0_5]
			Gvd0_5 = Gvz[down0_5]
			Gmu0_5 = Gm[up0_5]
			Gmd0_5 = Gm[down0_5]
			upout0_5 = Gvu0_5>100
			downout0_5 = Gvd0_5<-100
			totmom = np.absolute(np.sum(Gviu[upout]*Gmiu[upout]))+np.absolute(np.sum(Gvid[downout]*Gmid[downout]))
			totmom *= 1e10 # convert to solar mass *km/s

			totmom16 = np.absolute(np.sum(Gviu16[upout16]*Gmiu16[upout16]))+np.absolute(np.sum(Gvid16[downout16]*Gmid16[downout16]))
			totmom16 *= 1e10 # convert to solar mass *km/s

			totmom8 = np.absolute(np.sum(Gviu8[upout8]*Gmiu8[upout8]))+np.absolute(np.sum(Gvid8[downout8]*Gmid8[downout8]))
			totmom8 *= 1e10 # convert to solar mass *km/s

			totmom0_5 = np.absolute(np.sum(Gvu0_5[upout0_5]*Gmu0_5[upout0_5]))+np.absolute(np.sum(Gvd0_5[downout0_5]*Gmd0_5[downout0_5]))
			totmom0_5 *= 1e10 # convert to solar mass *km/s

			#totmomb = np.absolute(np.sum(Gvpb*Gmpb))*1e10




			#divide it by dL to get dM/dt:
			outflowrate = totmom/40./3.08567758e16*3.15569e7 #convert kpc to km, and s to yr so the unit is solar mass/yr
			outflowrate16 = totmom16/16./3.08567758e16*3.15569e7
			outflowrate8 = totmom8/8./3.08567758e16*3.15569e7
			outflowrate0_5 = totmom0_5/3.08567758e16*3.15569e7
			#outflowratepb = totmompb/3.08567758e16*3.15569e7
			snaplist = np.append(snaplist,Nsnapstring)
			outlist = np.append(outlist,outflowrate)
			outlist16 = np.append(outlist16,outflowrate16)
			outlist8 = np.append(outlist8, outflowrate8)
			outlist0_5 = np.append(outlist0_5, outflowrate0_5)
			
			#outlistpb = np.append(outlistpb, outflowratepb)
                        S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
			try:
				Sm = S['m']
				ms = np.sum(Sm)
				Sp = S['p']
				Sx = Sp[:,0]
				Sy = Sp[:,1]
				Sz = Sp[:,2]
				in1kpc = np.square(Sx)+np.square(Sy)+np.square(Sz)<1
				starmass1kpc = np.sum(Sm[in1kpc])
				del Sm, Sp, Sx, Sy, Sz
			except KeyError:
				ms = 0.0
				starmass1kpc = 0
			mslist=np.append(mslist,ms)
			sm1kpc=np.append(sm1kpc, starmass1kpc)
			gm1kpc=np.append(gm1kpc, gasmass1kpc)
			del S, Gvu0_5, Gmu0_5, upout0_5,Gvd0_5,Gmd0_5,downout0_5,Gviu8,Gmiu8,Gvid8,Gmid8,upout8,downout8,Gviu16,Gmiu16,Gvid16,Gmid16,upout16,downout16,Gviu,Gmiu,upout,Gmid,downout,G,Gp,Gv,Gm,Gvx,Gvy,Gvz,Gx,Gy,Gz
		spmname='outflow_'+runtodo+'.txt'
		spmfile=open(spmname,"w")
		spmfile.write('snapshot no.   ' + 'outflow (<20kpc)   '+'outflow (16-24kpc)    '+'outflow (8-12kpc)   '+'outflow (obs)      '+'stellar mass      '+'stellar mass (1kpc)       ' + 'gas mass (1kpc)       '+'\n')
		for ncount in range(len(snaplist)):
			spmfile.write(str(snaplist[ncount])+'    '+str(outlist[ncount])+'    '+str(outlist16[ncount])+'    '+str(outlist8[ncount])+'      '+str(outlist0_5[ncount])+'        '+str(mslist[ncount])+'              '+str(sm1kpc[ncount])+'         '+str(gm1kpc[ncount])+'\n')
		spmfile.close()


if wanted=='onlygamma':
        for runtodo in dirneed:
                print 'runtodo', runtodo
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr,  Fcal, iavesfr=outdirname(runtodo, Nsnap)
                G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix,havecr=1)
                Gegy = G['u']
                Gm = G['m']
                Gp = G['p']
                Gr = np.sqrt(Gp[:,0]*Gp[:,0]+Gp[:,1]*Gp[:,1]+Gp[:,2]*Gp[:,2])
                Grho = G['rho'] # in 10^10 Msun/kpc^3
                Gcregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                Gnism = Grho*1e10*2e33/2.9e64/1.1*6e23 #in cm^-3 assume: mean molecular weight=1.1 consider only nuclei
                tpi = 2e5/Gnism*250.0 #pi decay time in yr
                betapi=0.7 ##should be between 0.5-0.9 from Lacki 2011
                Lgammagev = Gcregy*0.25/tpi*2e53/3.2e7*betapi/3.0 #in erg/s   in proton calorimetric limit
                totLgammagev =np.sum(Lgammagev)
                print 'np.sum(Gcregy) in erg', np.sum(Gcregy)*2e53
                print 'np.sum(Gegy*Gm)', np.sum(Gegy*Gm)
                print 'np.amax(Gnism)', np.amax(Gnism)
                hist, bin_edges = np.histogram(Gnism, density=True, weights=G['m'])
                print 'np.average(Gnism)', np.average(Gnism)
                print 'In proton calorimetric limit'
                print ' Total gamma luminosity in solar (in log10)', np.log10(totLgammagev/3.85e33)





if wanted=='gamma':
        for runtodo in dirneed:
		print 'runtodo', runtodo
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr,  Fcal, iavesfr=outdirname(runtodo, Nsnap)
		G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix,havecr=1)
		S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix,havecr=1)
		Sage = S['age']
		Sm = S['m']
		Gegy = G['u']
		Gm = G['m']
		Gp = G['p']
		Gr = np.sqrt(Gp[:,0]*Gp[:,0]+Gp[:,1]*Gp[:,1]+Gp[:,2]*Gp[:,2])
		Grho = G['rho'] # in 10^10 Msun/kpc^3
		Gcregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
		Gnism = Grho*1e10*2e33/2.9e64/1.1*6e23 #in cm^-3 assume: mean molecular weight=1.1 consider only nuclei
		recentstar = Sage < 0.2
                aveSmass = np.average(S['m'])
                hist, bin_edges = np.histogram(Sage, bins=20)
                avesfr=hist[-1]*aveSmass/(bin_edges[1]-bin_edges[0])/0.98*10
		print 'avesfr', avesfr
		closer = Gr < 3.0
		print 'np.sum(Sm)', np.sum(Sm)
		tpi = 2e5/Gnism*250.0 #pi decay time in yr
		betapi=0.7 ##should be between 0.5-0.9 from Lacki 2011
		Lgammagev = Gcregy*0.25/tpi*2e53/3.2e7*betapi/3.0 #in erg/s   in proton calorimetric limit
		Lsfr =  3.8e-4 *avesfr*2e33/3.2e7*9e20
		totLgammagev =np.sum(Lgammagev)
		print 'estimated total supernova energy', np.sum(Sm)/0.4*1e10*0.0037*1e51
		print 'np.sum(Gcregy) in erg', np.sum(Gcregy)*2e53
		print 'np.sum(Gegy*Gm)', np.sum(Gegy*Gm)
		print 'np.amax(Gnism)', np.amax(Gnism)
		hist, bin_edges = np.histogram(Gnism, density=True, weights=G['m'])
		print 'np.average(Gnism)', np.average(Gnism)
		Lsfrideal = 3.8e-4 *iavesfr*2e33/3.2e7*9e20
		print 'Fsfr (ideal)', Lsfrideal/np.pi/4.0/(0.06*3.08e24)/(0.06*3.08e24)
		print 'In proton calorimetric limit'
		print ' Total gamma luminosity in solar (in log10)', np.log10(totLgammagev/3.85e33)
		print 'Fgamma/Fsfr', totLgammagev/Lsfr
		print 'with Fcal from LTQ fiducial model'
		print ' Total gamma luminosity in solar (in log10)', np.log10(totLgammagev*Fcal/3.85e33)
		print 'Fgamma/Fsfr', totLgammagev*Fcal/Lsfr
		print 'Fgamma/Fsfr (ideal)', totLgammagev*Fcal/Lsfrideal
		rneed=10
		fraction=0
		TotGcregy=np.sum(Gcregy)
		while (fraction>0.92) or (fraction<0.88):
			within = Gr<rneed
			fraction = np.sum(Gcregy[within])/TotGcregy
			rneed *= np.exp(1.0*(0.9-fraction))
			#print 'fraction, rneed', fraction, rneed
		hist, bin_edges = np.histogram(Gr,weights=Gcregy)
		plt.plot(bin_edges[:-1],hist, label=runtodo)
		plt.axvline(x=rneed)
	plt.ylabel(r'CR energy')
	plt.xlabel('kpc')
	plt.yscale('log')
	plt.xscale('log')
	plt.legend(loc='best')
	plt.savefig('histcr.pdf')
	plt.clf()


if wanted=='cosmic_raywrite':
        for runtodo in dirneed:
                snaplist=[]
                crlist=[]
		galist=[]
		smlist=[]
		aveGnisml=[]
		#for i in [1, 2]:
		#for i in [500]:
		print 'runtodo', runtodo
                for i in range(startno,Nsnap):
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep = outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
			print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)	
			Grho = G['rho']
       			Tb = G['u']
			cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
			Neb = G['ne']
			try:
				Sm = S['m']
			except KeyError:
				Sm = []
			#TrueTemp, converted_rho  = convertTemp(Tb, Neb, Grho, 1)
			#Gnism = Grho*1e10*2e33/2.9e64/1.1*6e23 #in cm^-3 assume: mean molecular weight=1.1 consider only nuclei
			Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
			tpi = 2e5/Gnism*250.0 #pi decay time in yr
			betapi = 0.7 ##should be between 0.5-0.9 from Lacki 2011
			#energylost=cregy/tpi*2e53*1e6
			#out=tpi/2<1e6
			#print 'len(out)/len(tot)', len(tpi[out])/float(len(tpi))
			#print 'crenergy in out', np.sum(cregy[out])
			Lgammagev = cregy/tpi*2e53/3.2e7*betapi/3.0 #in erg/s  
#			print 'gamma ray wo cut', np.sum(Lgammagev) 
#			for j in range(len(tpi)):
#				if tpi[j]/2<1e6:
#					Lgammagev[j]=cregy[j]/2e6*2e53/3.2e7*betapi/3.0
			if np.sum(cregy) >0:
				aveGnism=np.average(Gnism, weights=cregy)
			else:
				aveGnism=0
			snaplist=np.append(snaplist,Nsnapstring)
			crlist=np.append(crlist,np.sum(cregy))
			galist=np.append(galist,np.sum(Lgammagev))
			smlist=np.append(smlist,np.sum(Sm))
			aveGnisml=np.append(aveGnisml, aveGnism)
			print 'Gnsim', aveGnism
			print 'cosmic ray energy', np.sum(cregy)
			print 'gamma ray', np.sum(Lgammagev)
			print 'cr energy loss', np.sum(cregy)*2e53/2e5/3.2e7/250*aveGnism
			print 'cr energy inject', 0.8*2.9e40/17*24.3
			del G, S, Grho, Tb, cregy, Neb, Sm, Lgammagev, Gnism, tpi
		spmname='cregy_'+runtodo+'.txt'
		spmfile=open(spmname,"w")
		spmfile.write('snapshot no.   ' + 'cosmic ray energy     '+'gamma ray luminosity     '+' stellar mass  '+'Gnism     '+'\n')
		for ncount in range(len(snaplist)):
			spmfile.write(snaplist[ncount]+'    '+str(crlist[ncount])+'       '+str(galist[ncount])+'       '+str(smlist[ncount])+'       '+str(aveGnisml[ncount])+'\n')
		print 'file name', spmname
		spmfile.close()

if wanted=='gammatime':
	for runtodo in dirneed:
		rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
		spmname='cregy_'+runtodo+'.txt'
		spmfile=open(spmname,"r")
		spmfile.readline()
		dars = spmfile.readlines()

		snapno=[]
		cregy=[]
		galum=[]

		for line in dars:
			xsd = line.split()
			snapno = np.append(snapno, int(xsd[0]))
			cregy = np.append(cregy, float(xsd[1]))
			galum = np.append(galum, float(xsd[2]))
		spmfile.close()
		plt.plot(snapno*1e-3,galum,label=r'$\kappa_{\rm di} =$'+dclabel)
	plt.xlabel('Gyr')
	plt.title(runtitle)
	plt.ylabel('Gamma ray luminosity (erg/s)')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.savefig('galum.pdf')


if wanted=='crtime':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                spmname='cregy_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()

                snapno=[]
                cregy=[]
                galum=[]

                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        cregy = np.append(cregy, float(xsd[1]))
                        galum = np.append(galum, float(xsd[2]))
                spmfile.close()
                plt.plot(snapno,cregy*2e53,label='kappa='+dclabel+'; '+'time step= '+timestep)
        plt.xlabel('Myr')
        plt.title(runtitle)
        plt.ylabel('CR energy (erg)')
	plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/cregy.pdf')




if wanted=='gammasfr':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                spmname='cregy_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()
		print 'file name', spmname
                snapno=[]
                cregy=[]
                galum=[]
		smlist=[]
                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        cregy = np.append(cregy, float(xsd[1]))
                        galum = np.append(galum, float(xsd[2]))
			smlist = np.append(smlist, float(xsd[3]))
                spmfile.close()
		smlists=smlist[::snapsep]
		snapnos=snapno[::snapsep]
		cumgalum = np.cumsum(galum)
		cumgalums=cumgalum[::snapsep]
		sfr = (smlists[1:]-smlists[:-1])/(snapnos[1:]-snapnos[:-1])*1e4 
		avegalums = (cumgalums[1:]-cumgalums[:-1])/(snapnos[1:]-snapnos[:-1])
		#6.2e-4 comes from the different estimate of SNe rate in Lacki 2011 and in Chan 2015 DM paper
		plt.plot(snapnos[:-1],avegalums/(6.2e-4*sfr*2e33/3.2e7*9e20),label='kappa='+dclabel+'; '+'time step= '+timestep)
		print 'SFR', sfr
		print 'snapno', snapnos[:-1]
		print 'FgammaFsf', avegalums/(6.2e-4*sfr*2e33/3.2e7*9e20)
	plt.axhline(y=0.00023,ls='--',color='k')
        plt.xlabel('Myr')
	plt.xlim(xmax=Nsnap)
        plt.title(runtitle)
        plt.ylabel(r'$F_{\gamma}/F_{\rm SF}$')
	plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/galum_sfr_'+fmeat+'.pdf')
	plt.clf()

if wanted=='cumgamma':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                spmname='cregy_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()
                print 'file name', spmname
                snapno=[]
                cregy=[]
                galum=[]
                smlist=[]
                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        cregy = np.append(cregy, float(xsd[1]))
                        galum = np.append(galum, float(xsd[2]))
                        smlist = np.append(smlist, float(xsd[3]))
                spmfile.close()
                smlists=smlist[::snapsep]
                snapnos=snapno[::snapsep]
                cumgalum = np.cumsum(galum*3.15e7) #in erg
                cumgalums=cumgalum[::snapsep]/float(snapsep)
                plt.plot(snapnos,cumgalums,label='kappa='+dclabel+'; '+'time step= '+timestep)
        plt.xlabel('Myr')
	plt.ylabel('Total gamma ray energy in erg')
        plt.xlim(xmax=Nsnap)
        plt.title(runtitle)
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/cumgalum_'+fmeat+'.pdf')
        plt.clf()


if wanted=='crcooling':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                spmname='cregy_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()

                snapno=[]
                cregy=[]
                galum=[]

                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        cregy = np.append(cregy, float(xsd[1]))
                        galum = np.append(galum, float(xsd[2]))
                spmfile.close()
		crcooling = galum*3.0/0.7*2e5*250.0*3.2e7*7.51e-16
		snapnos=snapno[::snapsep]
		crcoolings=crcooling[::snapsep]  
                #plt.plot(snapnos,crcoolings,label=r'$\kappa_{\rm di} =$'+dclabel)
                plt.plot(snapnos,crcoolings,label='kappa='+dclabel+'; '+'time step= '+timestep)
        plt.xlabel('Myr')
        plt.title(runtitle)
        plt.ylabel('CR cooling rate (erg/s)')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/crcooling_'+runtodo+'.pdf')


if wanted=='sncretime':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
                Sage = S['age']
                aveSmass = np.average(S['m'])
                hist, bin_edges = np.histogram(Sage, bins=20)
                if havecr==0:
                        needlabel = slabel
                else:
                        needlabel = r' $\kappa_{\rm di}=$'+dclabel
		sncrerate=hist*aveSmass/(bin_edges[1]-bin_edges[0])/0.98*10/0.4*0.0037*1e51/3.16e7*0.1
		print 'sfr', hist*aveSmass/(bin_edges[1]-bin_edges[0])/0.98*10
                plt.plot(Nsnap*1e-3-bin_edges[1:]*0.98, sncrerate, label=needlabel)
                plt.xlabel('Gyr')
                plt.ylabel('CR power from SNe (erg/s)')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.title(runtitle)
        plt.savefig('sncre_'+runtodo+'.pdf')
        plt.clf()



if wanted=='crchange':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                spmname='cregy_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()

                snapno=[]
                cregy=[]
                galum=[]

                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        cregy = np.append(cregy, float(xsd[1]))
                        galum = np.append(galum, float(xsd[2]))
                spmfile.close()
		snapno=np.array(snapno[::50])
		cregy=np.array(cregy[::50])
                plt.plot(snapno[:-1],(cregy[1:]-cregy[:-1])/(snapno[1:]-snapno[:-1])*2e53/1e6/3e7,label='kappa='+dclabel+'; '+'time step= '+timestep)
        plt.xlabel('Myr')
        plt.title(runtitle)
        plt.ylabel('rate of change in CR energy (erg/s)')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/crchange_'+runtodo+'.pdf')


if wanted=='enchange':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                spmname='cregy_'+runtodo+'.txt'
                spmfile=open(spmname,"r")
                spmfile.readline()
                dars = spmfile.readlines()
                snapno=[]
                cregy=[]
                galum=[]
		smlist=[]
                for line in dars:
                        xsd = line.split()
                        snapno = np.append(snapno, int(xsd[0]))
                        cregy = np.append(cregy, float(xsd[1]))
                        galum = np.append(galum, float(xsd[2]))
			smlist = np.append(smlist, float(xsd[3]))
                spmfile.close()
                #S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
                #Sage = S['age']
                #aveSmass = np.average(S['m'])
                #hist, bin_edges = np.histogram(Sage, bins=20,weights=S['m'])
		smlist5=smlist[::5]
		snapno5=snapno[::5]
		#experiment 24.3/17 is different between Lacki 2011 and my DM paper in SNe rate.
		sncrerate = (smlist5[1:]-smlist5[:-1])/(snapno5[1:]-snapno5[:-1])*1e4/0.98/0.4*0.0037*1e51*0.1/3.2e7*24.3/17
                #sncrerate=hist/(bin_edges[1]-bin_edges[0])/0.98*10/0.4*0.0037*1e51*0.1/3.2e7
		#print 'bin_edges', bin_edges
		plt.plot(snapno5[:-1]/1e3, sncrerate, label=r'0.1$\dot{E}_{SN}$')
		snapno=snapno[::5]
		cregy=cregy[::5]
		galum=galum[::5]
		dcregy=(cregy[1:]-cregy[:-1])/(snapno[1:]-snapno[:-1])*2e53/1e6/3.2e7
		plt.plot(snapno[:-1]/1e3,dcregy,label=r'$\dot{E}_{\rm CR}$')
		if np.amin(dcregy)<0:
			plt.plot(snapno[:-1]/1e3,-dcregy,label=r'$-\dot{E}_{\rm CR}$')
                crcooling = galum*3.0/0.7*2e5*250.0*3.2e7*7.51e-16
                plt.plot(snapno/1e3,crcooling,label=r'$\dot{E}_{cooling}$')
	plt.xlabel('Gyr')
        plt.title(runtitle)
        plt.ylabel(r'$\dot{E}$(erg/s)')        
	plt.yscale('log')
	plt.xlim(xmax=Nsnap*1e-3)
        plt.legend(loc='best', fontsize=14)
        plt.savefig('enchange_'+runtodo+'.pdf')


if wanted=='checksnIItxt':
	for runtodo in dirneed:
		Nsnap=500 #does not have a role; arbitrary number
		rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, Nsnap)
		spmname='/home/tkc004/scratch/'+rundir+'/output/SNeIIheating.txt'
		spmfile=open(spmname,"r")
		spmfile.readline()
		dars = spmfile.readlines()

		timeinGyr=[]
		ntotal=[]

		for line in dars:
			xsd = line.split()
			timeinGyr = np.append(timeinGyr, float(xsd[0]))
			ntotal = np.append(ntotal, float(xsd[3]))
		spmfile.close()
		noofbins=100
		hist, bin_edges = np.histogram(timeinGyr, bins=noofbins, weights=ntotal)
		print 'sum', np.sum(ntotal)
		barwidth=(np.amax(timeinGyr)-np.amin(timeinGyr))/noofbins
		print 'barwidth', barwidth
		plt.plot(bin_edges[:-1], hist/barwidth*1e51/3.2e16, label=r'$\kappa_{\rm di}=$'+ dclabel)
		plt.xlabel('Gyr')
		plt.ylabel('Sne power (erg/s')
		plt.yscale('log')
		plt.legend(loc='best')
		plt.savefig('sn2egy.pdf')


if wanted=='crcsnap':
        for runtodo in dirneed:
                snaplist=[]
                crlist=[]
                galist=[]
                smlist=[]
                aveGnisml=[]
                #for i in [1, 2]:
                #for i in [500]:
                for i in range(Nsnap):
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			cregyl = G['cregyl']
			print 'CR energy loss rate (erg/s)', np.sum(cregyl*2e53/1e9/3.2e7)

if wanted=='crcgsnap':
	for runtodo in dirneed:
		snaplist=[]
		enclist=[]
		englist=[]
		enllist=[]
		endlist=[]
		enplist=[]
		prel=0
		preg=0
		prec=0
		pred=0
		prep=0
		for i in range(Nsnap):
			rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
			print 'the_snapdir', the_snapdir
			print 'Nsnapstring', Nsnapstring
			print 'havecr', havecr
			G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			cregyl = G['cregyl']
			cregyg = G['cregyg']
			cregy  = G['cregy']
			cregyd = G['cregyd']
			if havecr>4:
				cregyp = G['cregyp']
			cregygt=np.sum(cregyg)
			cregyt =np.sum(cregy)
			cregylt=np.sum(cregyl)
			cregydt=np.sum(cregyd)
			if havecr>4:
				cregydtp=np.sum(cregyp)
			eng = (cregygt-preg)/1e6/3.2e7*2e53
			enc = (cregyt-prec)/1e6/3.2e7*2e53
			enl = (cregylt-prel)*2e53/1e6/3.2e7
			end = (cregydt-pred)*2e53/1e6/3.2e7
			if havecr>4:
				enp = (cregydtp-prep)*2e53/1e6/3.2e7
			print 'CR energy loss rate (erg/s)', enl
			print 'CR energy gain rate (erg/s)', eng
			print 'CR energy change rate (erg/s)', enc
			print 'CR energy dt rate (erg/s)', end  #including adiabatic heating and streaming
			snaplist.append(i)
			enclist.append(enc)
			englist.append(eng)
			enllist.append(enl)
			endlist.append(end)
			if havecr>4:
				enplist.append(enp)
			preg = cregygt
			prec = cregyt
			prel = cregylt
			pred = cregydt
			if havecr>4:
				prep = cregydtp
		enclist=np.array(enclist)
		englist=np.array(englist)
		endlist=np.array(endlist)
		enllist=np.array(enllist)
		enplist=np.array(enplist)
		plt.plot(snaplist, enclist, label='CR energy change')
		plt.plot(snaplist, englist, label='CR energy gain')
		plt.plot(snaplist, endlist, label='CR energy dt')
		plt.plot(snaplist, enllist, label='CR energy loss')
		plt.plot(snaplist, englist+endlist+enllist, label='CR energy estimate')
		if havecr>4:
			plt.plot(snaplist, enplist, label='pre CR energy dt')
		#plt.yscale('log')
		plt.legend(loc='best')
		plt.xlabel('Myr', fontsize=25)
		plt.ylabel('dE/dt (erg/s)', fontsize=25)
		plt.savefig('cregsnap_'+fmeat+'.pdf')
		plt.clf()
		avesfr=1
		#Lsfr = 3.8e-4 *avesfr*2e33/3.2e7*9e20
		#above is the coefficient for Salpeter only; for Kroupa, the coefficient is 50% larger:
		Lsfr = 3.8e-4 *1.5*avesfr*2e33/3.2e7*9e20
		Lgammae = (enllist+endlist)/7.51e-16/2e5/3.2e7/250.0*0.7/3.0
		#From Guo 2008: 7.51e-16 is rate of total energy loss, including both hadronic and Coulomb
		# hadronic only: 5.86e-16; Coulumb only: 1.65e-16
		# decay rate = 7.51e-16*ne(cm^3)*rhocr(erg/cm^3) in s^-1
		# decay rate from Lacki et al. with the same unit: 6.25e-16 
                Lgamma = (enllist)/7.51e-16/2e5/3.2e7/250.0*0.7/3.0
		Lgammae_sfr = Lgammae/Lsfr
		Lgamma_sfr = Lgamma/Lsfr
		plt.plot(snaplist, np.absolute(Lgamma_sfr), label=r'Ours')
		plt.plot(snaplist, np.absolute(Lgammae_sfr), label='cooling-adiabatic')
		plt.yscale('log')
		plt.legend(loc='best')
                plt.xlabel('Myr', fontsize=25)
                plt.ylabel(r'$\frac{L_{\gamma}}{L_{\rm SF}}$', fontsize=30)
                plt.savefig('gammasfrsnapg_'+fmeat+'.pdf')
                plt.clf()
                
if wanted=='dirgammasfr':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
		sml=[]
		nsml=[]
		pretime=0
                for i in range(startno,Nsnap, snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
			cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
			if cosmo==1:
                                G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
                                S = readsnapcr(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
			else:
				G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
				S = readsnapcr(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        cregyl = G['cregyl']*1e10*solar_mass_in_g*km_in_cm*km_in_cm #in erg (originally in code unit: 1e10Msun*(km/s)^2)
                        cregyg = G['cregyg']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        cregy  = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
			try:
				header=S['header']
				timeneed=header[2]
				Smi=S['m']
				Sage=S['age']
				Sm = np.sum(Smi)*1e10 #in solar mass
				tcut=Sage>pretime
				Nsm = np.sum(Smi[tcut])*1e10
			except KeyError:
				Sm = 0.
				Nsm = 0.
				timeneed=0
                        cregygt=np.sum(cregyg)
                        cregyt =np.sum(cregy)
                        cregylt=np.sum(cregyl)
                        print 'CR energy loss (erg)', cregylt
                        print 'CR energy gain (erg)', cregygt
                        print 'CR energy change  (erg)', cregyt
                        snaplist.append(float(i))
                        enclist.append(cregyt)
                        englist.append(cregygt)
                        enllist.append(cregylt)
			sml.append(Sm)
			nsml.append(Nsm)
			pretime=timeneed
		sml=np.array(sml)
		Nsml=np.array(nsml)
		enclist=np.array(enclist)
		englist=np.array(englist)
		enllist=np.array(enllist)
		snaplist=np.array(snaplist)
		timel=np.array(snaplist*0.98*1e6) #in yr
		#above is the coefficient for Salpeter only; for Kroupa, the coefficient is 50% larger:
		avesfrl=(nsml[1:])/(timel[1:]-timel[:-1]) #in Msun/yr
		print 'avesfrl', avesfrl
		Lsfr = Kroupa_Lsf*avesfrl*solar_mass_in_g/sec_in_yr*cspeed_in_cm_s*cspeed_in_cm_s #times 1.5? 6.2e-4 for Kroupa 3.8e-4 for Salpeter
		print 'Lsfr', Lsfr
		print 'timel', timel
		Lgamma = (enllist[1:]-enllist[:-1])/((timel[1:]-timel[:-1])*sec_in_yr)/7.51e-16/pidecay_fac*betapi/nopi_per_gamma
		Lgamma_sfr = Lgamma/Lsfr
		print 'Lgamma', Lgamma
		#plt.plot(snaplist[1:], np.absolute(Lgamma_sfr[1:]), label='kappa = '+dclabel)
		plt.plot(timel[1:]/1e6, np.absolute(Lgamma_sfr), label=runtodo)
        plt.axhline(y=0.00023,ls='--',color='k')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.xlabel('Myr', fontsize=25)
	plt.ylabel(r'$\frac{L_{\gamma}}{L_{\rm SF}}$', fontsize=30)
	plt.savefig('CRplot/gammasfrsnap_'+fmeat+'.pdf')
	plt.clf()



if wanted=='dirsfr':
        for runtodo in dirneed:
                snaplist=[]
                avesfrl=[]
		avesfrnewl=[]
                presm = 0
                presnap = 0
		pretime=0
                for i in range(startno,Nsnap, snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
			cosmo=info['cosmo']
			color=info['color']
			print 'runtodo, color', runtodo, color
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
			if cosmo==1:
				S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
			else:
				S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        try:
                                Smi = S['m']
				Sage = S['age']
				Sm = np.sum(Smi)
				header=S['header']
				timeneed=header[2]
				#print 'time', header[2]
				#print 'Sage', Sage
                        except KeyError:
                                Sm = 0.
				Smi= 0.
				Sage = 0.
				timeneed=0.
                        snaplist.append(i)
			if i>presnap:
				if cosmo==1:
					snap2list, time2list=readtime(firever=2)
					tnow = np.interp(i,snap2list,time2list)*1e9
					pret = np.interp(presnap,snap2list,time2list)*1e9
					avesfr=(Sm-presm)*1e10/(tnow-pret)
				else:
					avesfr=(Sm-presm)*1e4/0.98/(i-presnap)
			else:
				avesfr=0
			if Sm>1e-9:	
				tcut=Sage>pretime
				Smnew = np.sum(Smi[tcut])
				avesfrnew = Smnew*10./(timeneed-pretime)
			else:
				avesfrnew=0.
                        presnap = i
                        presm = Sm
			pretime=timeneed
                        print 'avesfr', avesfr
			print 'avesfrnew', avesfrnew
			print 'Sm', Sm
                        avesfrl.append(avesfr)
			avesfrnewl.append(avesfrnew)
                avesfrl=np.array(avesfrl)
		avesfrnewl=np.array(avesfrnewl)
            #    plt.plot(snaplist, avesfrl, label=runtodo,color=color)
		plt.plot(snaplist, avesfrnewl, color=color)
        plt.yscale('log')
        plt.legend(loc='best')
        plt.xlabel('Myr', fontsize=25)
        plt.ylabel(r'${\rm SFR (M_{\odot}/yr)} $', fontsize=30)
        plt.savefig('CRplot/sfrsnap_'+fmeat+'.pdf')
        plt.clf()



if wanted=='dirsm':
        for runtodo in dirneed:
                tlist=[]
                sml=[]
                for i in range(startno,Nsnap, snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        if cosmo==1:
                                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
                        else:
                                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        try:
                                Sm = np.sum(S['m'])
                        except KeyError:
                                Sm = 0.
			if cosmo==1:
				snap2list, time2list=readtime(firever=2)
				tnow = np.interp(i,snap2list,time2list)*1e3
			else:
				tnow = 0.98*float(Nsnapstring)
                        print 'Sm', Sm
                        sml.append(Sm*1e10)
			tlist.append(tnow)
                sml=np.array(sml)
                plt.plot(tlist, sml, label=runtodo)
        plt.yscale('log')
        plt.legend(loc='best')
        plt.xlabel('Myr', fontsize=25)
        plt.ylabel(r'${\rm M_* (M_{\odot})} $', fontsize=30)
        plt.savefig('CRplot/smsnap_'+fmeat+'.pdf')
        plt.clf()





if wanted=='crasnap':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
		enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = 0
                for i in range(0,Nsnap, snapsep):
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        cregyl = G['cregyl']
                        cregyg = G['cregyg']
                        cregy  = G['cregy']
                        cregyd = G['cregyd']
                        if havecr>4:
                                cregyp = G['cregyp']
			if havecr>5:
				cregya = G['cregya']
                        try:
                                Sm = np.sum(S['m'])
                        except KeyError:
                                Sm = 0.
                        cregygt=np.sum(cregyg)
                        cregyt =np.sum(cregy)
                        cregylt=np.sum(cregyl)
                        cregydt=np.sum(cregyd)
                        if havecr>4:
                                cregydtp=np.sum(cregyp)
			if havecr>5:
				cregyat = np.sum(cregya)
                        eng = (cregygt)*2e53
                        enc = (cregyt)*2e53
                        enl = (cregylt)*2e53
                        end = (cregydt)*2e53
                        if havecr>4:
                                enp = (cregydtp)*2e53
                                print 'CR energy dtp', enp
			if havecr>5:
				ena = cregyat*2e53
				print 'CR energy dta', ena
                        snaplist.append(i)
                        enclist.append(enc)
                        englist.append(eng)
                        enllist.append(enl)
                        endlist.append(end)
                        if havecr>4:
                                enplist.append(enp)
			if havecr>5:
				enalist.append(ena)
                        if i>presnap:
                                avesfr=(Sm-presm)*1e4/0.98/(i-presnap)
                        else:
                                avesfr=0
                        print 'avesfr', avesfr
                        avesfrl.append(avesfr)
                avesfrl=np.array(avesfrl)
                enclist=np.array(enclist)
                englist=np.array(englist)
                endlist=np.array(endlist)
                enllist=np.array(enllist)
                enplist=np.array(enplist)
		enalist=np.array(enalist)
                plt.plot(snaplist, enclist, label='CR energy')
                plt.plot(snaplist, englist, label='SNe')
                if havecr > 4:
                        plt.plot(snaplist, endlist, label='Ad')
		#if havecr > 6:
		#	plt.plot(snaplist, enalist, label='Pure Ad')
		#if havecr > 7:
		#	plt.plot(snaplist, enplist, label='Other Ad')
                #plt.plot(snaplist, endlist, label='CR energy dt')
                plt.plot(snaplist, enllist, label='Loss')
                #plt.plot(snaplist, englist+endlist+enllist, label='CR energy estimate')
                if havecr>4:
                        plt.plot(snaplist, enclist-(englist+endlist+enllist), label='Extra')
                #plt.yscale('log')
		print 'havecr', havecr
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel('Myr', fontsize=25)
                plt.subplots_adjust(left=0.2,bottom=0.2, right=0.75)
                plt.ylabel(r'$E_{\rm tot}$ (erg)', fontsize=25)
                plt.savefig('CRplot/crasnap_'+fmeat+'.pdf')
                plt.clf()


if wanted=='crdelv':
        for runtodo in dirneed:
                snaplist=[]
                enplist=[]
		enalist=[]
                prep=0
		prea=0
                for i in range(Nsnap):
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        cregyp = G['cregyp']
			cregya = G['cregya']
			cregydtp=np.sum(cregyp)
			cregydta=np.sum(cregya)
			enp = (cregydtp-prep)*2e53/1e6/3.2e7
			ena = (cregydta-prea)*2e53/1e6/3.2e7	
                        snaplist.append(i)
			enplist.append(enp)
			enalist.append(ena)
                        prep = cregydtp
			prea = cregydta
                enplist=np.array(enplist)
		enalist=np.array(enalist)
                plt.plot(snaplist, enplist, label='CR energy change')
                plt.plot(snaplist, enalist, label='CR energy delv')
                plt.legend(loc='best')
                plt.xlabel('Myr', fontsize=25)
                plt.ylabel('dE/dt (erg/s)', fontsize=25)
                plt.savefig('credelv.pdf')
                plt.clf()


if wanted=='gasden':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
			dr = withinr/nogrid
			Gnism_in_cm_3l=[]
			radl =[]
			for irad in range(nogrid):
				cutxy = (Gx*Gx+Gy*Gy > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy < dr*(irad+1)*dr*(irad+1))
				cutz = np.absolute(Gz-med)< maxlength/2.
				cut = cutxy*cutz
				Nebcut = Neb[cut]
				Gmcut = Gm[cut]	
				Gm_in_g = Gmcut*1e10*2e33
				shellvol_in_cm3 = np.pi*maxlength*(-np.power(dr*irad,2)+np.power(dr*(irad+1),2))*3.086e21*3.086e21*3.086e21
				Grho_in_g_cm_3 = Gm_in_g/shellvol_in_cm3
				protonmass_in_g = 1.67e-24
				Gnism_in_cm_3 = np.sum((0.78+0.22*Nebcut*0.76)/protonmass_in_g*Grho_in_g_cm_3)
				Gnism_in_cm_3l = np.append(Gnism_in_cm_3l, Gnism_in_cm_3)
				radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, Gnism_in_cm_3l, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$n_{\rm ISM} [{\rm cm^{-3}}]$')
        plt.legend(loc='best')
	plt.title('cylinder centered at disk '+str(med)+' kpc above z plane with a height ' + str(maxlength) + ' kpc')
        plt.savefig('gasdensity_'+fmeat+'_r.pdf')
        plt.clf()

if wanted=='gmr':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        Gm_in_sunl=[]
                        radl =[]
                        for irad in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy < dr*irad*dr*irad) 
                                cutz = np.absolute(Gz-med)< maxlength/2.
				cut = cutxy*cutz
                                Gmcut = Gm[cut]
				Gm_in_sun = Gmcut*1e10
                                Gm_in_sunl = np.append(Gm_in_sunl, np.sum(Gm_in_sun))
                                radl = np.append(radl, dr*irad)
                        plt.plot(radl, Gm_in_sunl, label=runtodo)
	plt.yscale('log')
        plt.xlabel('r [kpc]')
        plt.ylabel(r'enclosed $M_{\rm g} [M_{\odot}]$')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk '+str(med)+' kpc above z plane with a height ' + str(maxlength) + ' kpc')
        plt.savefig('gasmass_'+fmeat+'_r_zmed'+str(med)+'.pdf')
        plt.clf()

if wanted=='crer':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        crel=[]
                        radl =[]
                        for irad in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy < dr*irad*dr*irad)
                                cutz = np.absolute(Gz-med)< maxlength/2.
                                cut = cutxy*cutz
                                crecut = cregy[cut]
                                crel = np.append(crel, np.sum(crecut))
                                radl = np.append(radl, dr*irad)
                        plt.plot(radl, crel, label=runtodo)
        plt.yscale('log')
        plt.xlabel('r [kpc]')
        plt.ylabel(r'enclosed $E_{\rm cr} [{\rm 10^{10}M_{\odot}km^2/s^2}]$')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk '+str(med)+' kpc above z plane with a height ' + str(maxlength) + ' kpc')
        plt.savefig('cre_'+fmeat+'_r_zmed'+str(med)+'.pdf')
        plt.clf()


if wanted=='gasdenz':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dz = maxlength/nogrid
                        Gnism_in_cm_3l=[]
                        zl =[]
                        for iz in range(nogrid):
                                cutxy = Gx*Gx+Gy*Gy < withinr
                                cutz = (Gz> dz*iz-maxlength/2.) & (Gz<dz*(iz+1)-maxlength/2.)
                                cut = cutxy*cutz
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gm_in_g = Gmcut*1e10*2e33
                                shellvol_in_cm3 = np.pi*dz*np.power(withinr,2)*3.086e21*3.086e21*3.086e21
                                Grho_in_g_cm_3 = Gm_in_g/shellvol_in_cm3
                                protonmass_in_g = 1.67e-24
                                Gnism_in_cm_3 = np.sum((0.78+0.22*Nebcut*0.76)/protonmass_in_g*Grho_in_g_cm_3)
                                Gnism_in_cm_3l = np.append(Gnism_in_cm_3l, Gnism_in_cm_3)
                                zl = np.append(zl, dz*(iz+0.5)-maxlength/2.)
                        plt.plot(zl, Gnism_in_cm_3l, label=runtodo)
        plt.xlabel('z [kpc]')
        plt.ylabel(r'$n_{\rm ISM} [{\rm cm^{-3}}]$')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk with a radius ' + str(withinr) + ' kpc')
        plt.savefig('gasdensity_'+fmeat+'_z.pdf')
        plt.clf()


if wanted=='gmz':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dz = maxlength/nogrid
                        Gm_in_sunl=[]
                        zl =[]
                        for iz in range(nogrid):
                                cutxy = Gx*Gx+Gy*Gy < withinr
                                cutz = np.absolute(Gz)<dz*(iz+1)
                                cut = cutxy*cutz
                                Gmcut = Gm[cut]
                                Gm_in_sun = Gmcut*1e10
                                Gm_in_sunl = np.append(Gm_in_sunl, np.sum(Gm_in_sun))
                                zl = np.append(zl, dz*(iz+1))
                        plt.plot(zl, Gm_in_sunl, label=runtodo)
	plt.yscale('log')
        plt.xlabel('z [kpc]')
        plt.ylabel(r'enclosed $M_{\rm gas} [ M_\odot]$')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk with a radius ' + str(withinr) + ' kpc')
        plt.savefig('gasmass_'+fmeat+'_z.pdf')
        plt.clf()


if wanted=='crez':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dz = maxlength/nogrid
                        crel=[]
                        zl =[]
                        for iz in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy < withinr*withinr)
				cutz = np.absolute(Gz)<dz*iz
                                cut = cutxy*cutz
                                crecut = cregy[cut]
                                crel = np.append(crel, np.sum(crecut))
                                zl = np.append(zl, dz*iz)
                        plt.plot(zl, crel, label=runtodo)
        plt.yscale('log')
        plt.xlabel('z [kpc]')
        plt.ylabel(r'enclosed $E_{\rm cr} [{\rm 10^{10}M_{\odot}km^2/s^2}]$')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk with a radius ' + str(withinr) + ' kpc')
        plt.savefig('cre_'+fmeat+'_z.pdf')
        plt.clf()




if wanted=='nismcre':
        for runtodo in dirneed:
                timel=[]
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        timeneed=i*0.98*1e9 #in yr
                        Grho = G['rho'] #1e10Msun per kpc^3
                        cregy = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm #original cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        timel=np.append(timel,timeneed)
                Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/proton_mass_in_g*Grho*1e10*solar_mass_in_g/kpc_in_cm/kpc_in_cm/kpc_in_cm
                #tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                LogGnism = np.log10(Gnism_in_cm_3)
                LogGnxaxis = np.linspace(-4,4,num=nogrid)
                dx =LogGnxaxis[1]-LogGnxaxis[0]
                Ecrl = []
                for inism in range(nogrid-1):
                        cutg = (LogGnism > LogGnxaxis[inism]) & (LogGnism < LogGnxaxis[inism+1])
                        Ecrcut = cregy[cutg]
                        Ecrl = np.append(Ecrl, np.sum(Ecrcut))
                plt.plot((LogGnxaxis[1:]+LogGnxaxis[:-1])/2.,Ecrl/dx, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (n_{\rm ISM}[{\rm cm^{-3}}])$')
        plt.ylabel(r'$\mathrm{d} E_{cr}/\mathrm{d} \log (n_{\rm ISM})[{\rm erg}]$')
        plt.savefig('CRplot/nismcr_'+fmeat+'.pdf')
        plt.clf()



if wanted=='nismcumcre':
        for runtodo in dirneed:
                timel=[]
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        timeneed=i*0.98*1e9 #in yr
                        Grho = G['rho'] #1e10Msun per kpc^3
                        cregy = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm #original cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        timel=np.append(timel,timeneed)
                Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/proton_mass_in_g*Grho*1e10*solar_mass_in_g/kpc_in_cm/kpc_in_cm/kpc_in_cm
                #tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                LogGnism = np.log10(Gnism_in_cm_3)
                LogGnxaxis = np.linspace(-4,4,num=nogrid)
                dx =LogGnxaxis[1]-LogGnxaxis[0]
                Ecrl = []
                for inism in range(nogrid-1):
                        cutg = (LogGnism > LogGnxaxis[inism]) 
                        Ecrcut = cregy[cutg]
                        Ecrl = np.append(Ecrl, np.sum(Ecrcut))
                plt.plot((LogGnxaxis[1:]+LogGnxaxis[:-1])/2.,Ecrl, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (n_{\rm ISM}[{\rm cm^{-3}}])$')
        plt.ylabel(r'$E_{cr} (>n_{\rm ISM})[{\rm erg}]$')
        plt.savefig('CRplot/nismcumcr_'+fmeat+'.pdf')
        plt.clf()




if wanted=='credecaytime':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        logtpi = np.log10(tpi_in_yr)
                        logtpixaxis = np.linspace(5,14,num=nogrid)
			dx = logtpixaxis[1]-logtpixaxis[0]
                        cregy_in_ergl = []
                        for inism in range(nogrid-1):
                                cutt = (logtpi > logtpixaxis[inism]) & (logtpi < logtpixaxis[inism+1])
                                cregy_in_ergcut = cregy_in_erg[cutt]
                                cregy_in_ergl = np.append(cregy_in_ergl, np.sum(cregy_in_ergcut))
                        plt.plot((logtpixaxis[1:]+logtpixaxis[:-1])/2.,cregy_in_ergl/dx, label=runtodo)
        plt.legend(loc='best')
	plt.yscale('log')
        plt.xlabel(r'$\log (t_{\rm \pi}[{\rm yr}])$')
        #plt.ylabel(r'$\frac{\mathrm{d} E_{\rm cr}}{\mathrm{d} n_{\rm ISM}}\Delta \log n_{\rm ISM}[{\rm erg/cm^3}]$')
        plt.ylabel(r'$\mathrm{d} E_{\rm cr}/\mathrm{d} \log (t_{\rm \pi})[{\rm erg}]$')
        plt.savefig('credecaytime_'+fmeat+'.pdf')
        plt.clf()


if wanted=='crecumdecaytime':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        logtpi = np.log10(tpi_in_yr)
                        logtpixaxis = np.linspace(5,14,num=nogrid)
                        cregy_in_ergl = []
                        for inism in range(nogrid-1):
                                cutt =  (logtpi > logtpixaxis[inism])
                                cregy_in_ergcut = cregy_in_erg[cutt]
                                cregy_in_ergl = np.append(cregy_in_ergl, np.sum(cregy_in_ergcut))
                        plt.plot((logtpixaxis[1:]+logtpixaxis[:-1])/2.,cregy_in_ergl, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (t_{\rm \pi}[{\rm yr}])$')
        #plt.ylabel(r'$\frac{\mathrm{d} E_{\rm cr}}{\mathrm{d} n_{\rm ISM}}\Delta \log n_{\rm ISM}[{\rm erg/cm^3}]$')
        plt.ylabel(r'$E_{\rm cr}(>t_{\rm \pi})[{\rm erg}]$')
        plt.savefig('crecumdecaytime_'+fmeat+'.pdf')
        plt.clf()


if wanted=='cregamma':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
			print 'np.sum(Lgammagev)', np.sum(Lgammagev)
                        logLgamma = np.log10(Lgammagev)
                        loggammaxaxis = np.linspace(25,42,num=nogrid)
			dx = loggammaxaxis[1]-loggammaxaxis[0]
                        cregy_in_ergl = []
                        for inism in range(nogrid-1):
                                cutt = ( logLgamma> loggammaxaxis[inism]) & (logLgamma < loggammaxaxis[inism+1])
                                cregy_in_ergcut = cregy_in_erg[cutt]
                                cregy_in_ergl = np.append(cregy_in_ergl, np.sum(cregy_in_ergcut))
                        plt.plot((loggammaxaxis[1:]+loggammaxaxis[:-1])/2.,cregy_in_ergl/dx, label=runtodo)
        plt.legend(loc='best')
	plt.yscale('log')
        plt.xlabel(r'$\log (L_{\rm \gamma}[{\rm erg/s}])$')
        #plt.ylabel(r'$\frac{\mathrm{d} E_{\rm cr}}{\mathrm{d} n_{\rm ISM}}\Delta \log n_{\rm ISM}[{\rm erg/cm^3}]$')
        plt.ylabel(r'$\mathrm{d} E_{\rm cr}\mathrm{d} \log (L_{\gamma})[{\rm erg}]$')
        plt.savefig('cregamma_'+fmeat+'.pdf')
        plt.clf()

if wanted=='nismgamma':
        for runtodo in dirneed:
                timel=[]
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        timeneed=i*0.98*1e9 #in yr
                        Grho = G['rho'] #1e10Msun per kpc^3
                        cregy = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm #original cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        cregyl = G['cregyl']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        timel=np.append(timel,timeneed)
                Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/proton_mass_in_g*Grho*1e10*solar_mass_in_g/kpc_in_cm/kpc_in_cm/kpc_in_cm
                #tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                Eloss=cregyl/7.51e-16/pidecay_fac*betapi/nopi_per_gamma
                Lgamma=np.absolute((Eloss)/float(Nsnap)/0.98e6/sec_in_yr)
                LogGnism = np.log10(Gnism_in_cm_3)
                LogGnxaxis = np.linspace(-4,4,num=nogrid)
		dx =LogGnxaxis[1]-LogGnxaxis[0]
                Lgammal = []
                print 'Eloss', Eloss
		for inism in range(nogrid-1):
			cutg = (LogGnism > LogGnxaxis[inism]) & (LogGnism < LogGnxaxis[inism+1])
			Lgammacut = Lgamma[cutg]
			Lgammal = np.append(Lgammal, np.sum(Lgammacut))
		plt.plot((LogGnxaxis[1:]+LogGnxaxis[:-1])/2.,Lgammal/dx, label=runtodo)
        plt.legend(loc='best')
	plt.yscale('log')
        plt.xlabel(r'$\log (n_{\rm ISM}[{\rm cm^{-3}}])$')
        plt.ylabel(r'$\mathrm{d} L_{\gamma}/\mathrm{d} \log (n_{\rm ISM})[{\rm erg/s}]$')
        plt.savefig('CRplot/nismgamma_'+fmeat+'.pdf')
        plt.clf()

if wanted=='nismcumgamma':
        for runtodo in dirneed:
		timel=[]
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			timeneed=i*0.98*1e9 #in yr
			Grho = G['rho'] #1e10Msun per kpc^3
			cregy = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm #original cosmic ray energy in 1e10Msun km^2/sec^2
			Neb = G['ne']
			cregyl = G['cregyl']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
			timel=np.append(timel,timeneed)
		Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/proton_mass_in_g*Grho*1e10*solar_mass_in_g/kpc_in_cm/kpc_in_cm/kpc_in_cm
		#tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
		#Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                Eloss=cregyl/7.51e-16/pidecay_fac*betapi/nopi_per_gamma
		Lgamma=np.absolute((Eloss)/float(Nsnap)/0.98e6/sec_in_yr)
		LogGnism = np.log10(Gnism_in_cm_3)
		LogGnxaxis = np.linspace(-4,4,num=nogrid)
		Lgammal = []
		print 'Eloss', Eloss
		for inism in range(nogrid-1):
			cutg = (LogGnism > LogGnxaxis[inism])
			Lgammacut = Lgamma[cutg]
                        Lgammal = np.append(Lgammal, np.sum(Lgammacut))
		print 'Lgammal', Lgammal
		plt.plot((LogGnxaxis[1:]+LogGnxaxis[:-1])/2.,Lgammal, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (n_{\rm ISM}[{\rm cm^{-3}}])$')
        plt.ylabel(r'$L_{\gamma}(>n_{\rm ISM})[{\rm erg/s}]$')
        plt.savefig('CRplot/nismcumgamma_'+fmeat+'.pdf')
        plt.clf()


if wanted=='gammadecaytime':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        logtpi = np.log10(tpi_in_yr)
                        logtpixaxis = np.linspace(5,14,num=nogrid)
                        dx = logtpixaxis[1]-logtpixaxis[0]
                        Lgammagevl = []
                        for inism in range(nogrid-1):
                                cutt = (logtpi > logtpixaxis[inism]) & (logtpi < logtpixaxis[inism+1])
                                Lgammagevcut = Lgammagev[cutt]
                                Lgammagevl = np.append(Lgammagevl, np.sum(Lgammagevcut))
                        plt.plot((logtpixaxis[1:]+logtpixaxis[:-1])/2.,Lgammagevl/dx, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (t_{ \pi}[{\rm yr}])$')
        #plt.ylabel(r'$\frac{\mathrm{d} E_{\rm cr}}{\mathrm{d} n_{\rm ISM}}\Delta \log n_{\rm ISM}[{\rm erg/cm^3}]$')
        plt.ylabel(r'$\mathrm{d} L_{\gamma} /\mathrm{d} \log (t_{ \pi})[{\rm erg/s}]$')
        plt.savefig('gammadecaytime_'+fmeat+'.pdf')
        plt.clf()


if wanted=='gammacumdecaytime':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        logtpi = np.log10(tpi_in_yr)
                        logtpixaxis = np.linspace(5,14,num=nogrid)
                        Lgammagevl = []
                        for inism in range(nogrid-1):
                                cutt = (logtpi > logtpixaxis[inism])
                                Lgammagevcut = Lgammagev[cutt]
                                Lgammagevl = np.append(Lgammagevl, np.sum(Lgammagevcut))
                        plt.plot((logtpixaxis[1:]+logtpixaxis[:-1])/2.,Lgammagevl, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (t_{ \pi}[{\rm yr}])$')
        #plt.ylabel(r'$\frac{\mathrm{d} E_{\rm cr}}{\mathrm{d} n_{\rm ISM}}\Delta \log n_{\rm ISM}[{\rm erg/cm^3}]$')
        plt.ylabel(r'$L_{\gamma}(>t_{ \pi})[{\rm erg/s}]$')
        plt.savefig('gammacumdecaytime_'+fmeat+'.pdf')
        plt.clf()



if wanted=='gmnism':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
			Gm = G['m']
                        #cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        #tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        #betapi = 0.7 ##should be between 0.5-0.9 from Lacki 2011 #fraction of pi has enough energy (>1GeV)
                        #sec_in_yr = 3.2e7
                        #nopi_per_gamma = 3.0
			Gm_in_sun = Gm*1e10 
                        #cm_in_km = 1e5
                        #cregy_in_erg = cregy*solar_mass_in_g*1e10*cm_in_km*cm_in_km
                        #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        #print 'np.sum(Lgammagev)', np.sum(Lgammagev)
                        LogGnism = np.log10(Gnism_in_cm_3)
                        LogGnxaxis = np.linspace(-4,4,num=nogrid)
                        Gml = []
                        dx = LogGnxaxis[1]-LogGnxaxis[0]
                        for inism in range(nogrid-1):
                                cutg = (LogGnism > LogGnxaxis[inism]) & (LogGnism < LogGnxaxis[inism+1])
                                Gmcut = Gm_in_sun[cutg]
                                Gml = np.append(Gml, np.sum(Gmcut))
                        plt.plot((LogGnxaxis[1:]+LogGnxaxis[:-1])/2.,Gml/dx, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (n_{\rm ISM}[{\rm cm^{-3}}])$')
        plt.ylabel(r'$\mathrm{d} M_{\rm gas}/\mathrm{d} \log (n_{\rm ISM})[M_\odot]$')
        plt.savefig('gmnism_'+fmeat+'.pdf')
        plt.clf()


if wanted=='nismcumgm':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Grho = G['rho']
                        Gm = G['m']
                        #cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        #tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        #betapi = 0.7 ##should be between 0.5-0.9 from Lacki 2011 #fraction of pi has enough energy (>1GeV)
                        #sec_in_yr = 3.2e7
                        #nopi_per_gamma = 3.0
                        Gm_in_sun = Gm*1e10
                        #cm_in_km = 1e5
                        #cregy_in_erg = cregy*solar_mass_in_g*1e10*cm_in_km*cm_in_km
                        #Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        #print 'np.sum(Lgammagev)', np.sum(Lgammagev)
                        LogGnism = np.log10(Gnism_in_cm_3)
                        LogGnxaxis = np.linspace(-4,4,num=nogrid)
                        Gml = []
                        dx = LogGnxaxis[1]-LogGnxaxis[0]
                        for inism in range(nogrid-1):
                                cutg = (LogGnism > LogGnxaxis[inism]) 
                                Gmcut = Gm_in_sun[cutg]
                                Gml = np.append(Gml, np.sum(Gmcut))
                        plt.plot((LogGnxaxis[1:]+LogGnxaxis[:-1])/2.,Gml, label=runtodo)
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(r'$\log (n_{\rm ISM}[{\rm cm^{-3}}])$')
        plt.ylabel(r'$M_{\rm gas}(>n_{\rm ISM})[M_\odot]$')
        plt.savefig('CRplot/nismcumgm_'+fmeat+'.pdf')
        plt.clf()



if wanted=='gasdensph':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
			Gy = Gp[:,1]
                        dr = withinr/nogrid
                        Gnism_in_cm_3l=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gm_in_g = Gmcut*1e10*2e33
                                shellvol_in_cm3 = 4.0/3.0*np.pi*(-np.power(dr*irad,3)+np.power(dr*(irad+1),3))*3.086e21*3.086e21*3.086e21
                                Grho_in_g_cm_3 = Gm_in_g/shellvol_in_cm3
                                protonmass_in_g = 1.67e-24
                                Gnism_in_cm_3 = np.sum((0.78+0.22*Nebcut*0.76)/protonmass_in_g*Grho_in_g_cm_3)
                                Gnism_in_cm_3l = np.append(Gnism_in_cm_3l, Gnism_in_cm_3)
                                radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, Gnism_in_cm_3l, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$n_{\rm ISM} [{\rm cm^{-3}}]$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('spherical shell centered at the center of the disk')
        plt.savefig('gasdensity_'+fmeat+'_sph.pdf')
        plt.clf()


if wanted=='credensph':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        cregydenl=[]
                        radl =[]
			print 'cregy', cregy
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                cregycut = cregy[cut]
                                shellvol_in_kpc3 = 4.0/3.0*np.pi*(-np.power(dr*irad,3)+np.power(dr*(irad+1),3))
                                cregyden = cregycut/shellvol_in_kpc3
                                cregydenl = np.append(cregydenl, np.sum(cregyden))
                                radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, cregydenl, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$e_{\rm CR} [{\rm 10^{10}M_{\odot}km^2/s^2/kpc^3}]$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('spherical shell centered at the center of the disk')
        plt.savefig('credensity_'+fmeat+'_sph.pdf')
        plt.clf()

if wanted=='gammadensph':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        dr = withinr/nogrid
                        Lgdenl=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                Nebcut = Neb[cut]
                                Lgammacut = Lgammagev[cut]
                                shellvol_in_kpc3 = 4.0/3.0*np.pi*(-np.power(dr*irad,3)+np.power(dr*(irad+1),3))
                                Lgden = Lgammacut/shellvol_in_kpc3
				Lgdenl = np.append(Lgdenl, np.sum(Lgden))
                                radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, Lgdenl, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$l_{\gamma} [{\rm erg/s/kpc^{3}}]$')
	plt.yscale('log')
        plt.legend(loc='best')
        plt.title('spherical shell centered at the center of the disk')
        plt.savefig('gammadensity_'+fmeat+'_sph.pdf')
        plt.clf()

if wanted=='Gmencsph':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        Gmenc_in_sunl=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gmenc_in_sun = Gmcut*1e10
                                Gmenc_in_sunl = np.append(Gmenc_in_sunl, np.sum(Gmenc_in_sun))
                                radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, Gmenc_in_sunl, label='kappa='+dclabel+'; '+'time step= '+timestep)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$M_{\rm gas} [ M_{\odot}]$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('Acummulative gas mass within a sphere centered at the center of the disk')
        plt.savefig('CRplot/Gmenc_'+fmeat+'_sph.pdf')
        plt.clf()


if wanted=='cresph':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        crel=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad)*dr*(irad))
                                crecut = cregy[cut]
                                crel = np.append(crel, np.sum(crecut))
                                radl = np.append(radl, dr*(irad))
                        plt.plot(radl, crel, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$E_{rm cr} [{\rm 10^{10}M_{\odot}km^2/s^2}]$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.title('Cosmic ray energy within a sphere centered at the center of the disk')
        plt.savefig('cre_'+fmeat+'_sph.pdf')
        plt.clf()



if wanted=='gammasph':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        dr = withinr/nogrid
                        Lgammal=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                Nebcut = Neb[cut]
                                Lgammacut = Lgammagev[cut]
                                Lgammal = np.append(Lgammal, np.sum(Lgammacut))
                                radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, Lgammal, label='kappa='+dclabel+'; '+'time step= '+timestep)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$L_{\gamma} [{\rm erg/s}]$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.title('Gamma ray luminosity within a sphere centered at the center of the disk')
        plt.savefig('CRplot/gamma_'+fmeat+'_sph.pdf')
        plt.clf()

if wanted=='decaytimesph':
	icolor=0
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr,timestep=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        tpi_in_yrl=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gm_in_g = Gmcut*1e10*2e33
                                shellvol_in_cm3 = 4.0/3.0*np.pi*(np.power(dr*irad,3))*3.086e21*3.086e21*3.086e21
                                Grho_in_g_cm_3 = Gm_in_g/shellvol_in_cm3
                                protonmass_in_g = 1.67e-24
                                Gnism_in_cm_3 = np.sum((0.78+0.22*Nebcut*0.76)/protonmass_in_g*Grho_in_g_cm_3)
				tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                                tpi_in_yrl = np.append(tpi_in_yrl, np.sum(tpi_in_yr))
                                radl = np.append(radl, dr*irad)
                        plt.plot(radl, tpi_in_yrl, label=runtodo, color=colortable[icolor])
			print 'icolor', icolor
		icolor +=1
	diffusioncoefficient = 3e28 #in cm^2/s
	esctime = radl*radl*3.086e21*3.086e21/diffusioncoefficient/3.2e7
	plt.plot(radl, esctime*10., label=r'Escape time for $\kappa_{di} = 3\times 10^{27}$', ls='dashed', color=colortable[1])
	plt.plot(radl, esctime, label=r'Escape time for $\kappa_{di} = 3\times 10^{28}$', ls='dashed', color=colortable[2])
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$t_\pi [{\rm yr}]$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('Within sphere centered at the center of the disk')
        plt.savefig('decaytime_'+fmeat+'_sph.pdf')
        plt.clf()


if wanted=='decayratiosph':
        icolor=0
	diffusiontable=[0, 3e27, 3e28]
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        tpi_in_yrl=[]
                        radl =[]
                        for irad in range(nogrid):
                                cut = (Gx*Gx+Gy*Gy+Gz*Gz < dr*(irad+1)*dr*(irad+1))
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gm_in_g = Gmcut*1e10*2e33
                                shellvol_in_cm3 = 4.0/3.0*np.pi*(np.power(dr*irad,3))*3.086e21*3.086e21*3.086e21
                                Grho_in_g_cm_3 = Gm_in_g/shellvol_in_cm3
                                protonmass_in_g = 1.67e-24
                                Gnism_in_cm_3 = np.sum((0.78+0.22*Nebcut*0.76)/protonmass_in_g*Grho_in_g_cm_3)
                                tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                                tpi_in_yrl = np.append(tpi_in_yrl, np.sum(tpi_in_yr))
                                radl = np.append(radl, dr*irad)
			diffusioncoefficient = diffusiontable[icolor] #in cm^2/s
			esctime = radl*radl*3.086e21*3.086e21/diffusioncoefficient/3.2e7
                 	if icolor>0:
			       plt.plot(radl, esctime/(tpi_in_yrl+esctime), label=runtodo, color=colortable[icolor])
                        print 'icolor', icolor
                icolor +=1
        plt.xlabel('r [kpc]')
        plt.ylabel(r'fraction of decayed CR $\sim 1/(1+t_{\pi}/t_{\rm esc})$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('Within sphere centered at the center of the disk')
        plt.savefig('decayratio_'+fmeat+'_sph.pdf')
	plt.clf()


if wanted=='gammar':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        dr = withinr/nogrid
                        Lgammal=[]
                        radl =[]
                        for irad in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy < dr*irad*dr*irad)
                                cutz = np.absolute(Gz-med)< maxlength/2.
                                cut = cutxy*cutz
                                Nebcut = Neb[cut]
                                Lgammacut = Lgammagev[cut]
                                Lgammal = np.append(Lgammal, np.sum(Lgammacut))
                                radl = np.append(radl, dr*(irad))
                        plt.plot(radl, Lgammal, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'enclosed $L_{\gamma} [{\rm erg/s}]$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk '+str(med)+' kpc above z plane with a height ' + str(maxlength) + ' kpc')
        plt.savefig('gamma_'+fmeat+'_r_zmed'+str(med)+'.pdf')
        plt.clf()

if wanted=='gammaz':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr=outdirname(runtodo, i)
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21
                        tpi_in_yr = 2e5/Gnism_in_cm_3*250.0 #pi decay time in yr
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_yr/sec_in_yr*betapi/nopi_per_gamma #in erg/s
                        dz = maxlength/nogrid
                        Lgammal=[]
                        zl =[]
                        for iz in range(nogrid):
                                cutxy = Gx*Gx+Gy*Gy < withinr
                                cutz = np.absolute(Gz)<dz*(iz+1)
                                cut = cutxy*cutz
                                Nebcut = Neb[cut]
                                Lgammacut = Lgammagev[cut]
                                Lgammal = np.append(Lgammal, np.sum(Lgammacut))
                                zl = np.append(zl, dz*(iz))
                        plt.plot(zl, Lgammal, label=runtodo)
        plt.xlabel('z [kpc]')
        plt.ylabel(r'enclosed $L_{\gamma} [{\rm erg/s}]$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.title('cylinder centered at disk with a radius ' + str(withinr) + ' kpc')
        plt.savefig('gamma_'+fmeat+'_z.pdf')
        plt.clf()

if wanted=='crdensol': #cosmic ray energy density at solar circle (observation ~ 1 eV/cm^3)
        for runtodo in dirneed:
		snaplist=[]
		credenlist=[]
                for i in range(startno, Nsnap, snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
			cregy_in_eV = cregy_in_erg*erg_in_eV
			solar_radius = 8.
			wdisk=2.
			insol = solar_radius+wdisk/2.
			outsol = solar_radius-wdisk/2.
			hdisk = 2.
			cutxy = (Gx*Gx+Gy*Gy < insol*insol) & (Gx*Gx+Gy*Gy > outsol*outsol) 
			cutz = np.absolute(Gz) < hdisk/2.
			cut = cutxy*cutz
			cregy_in_eV_cut = np.sum(cregy_in_eV[cut])
			creden_in_eV_per_cm3 = cregy_in_eV_cut/(np.pi*(insol*insol-outsol*outsol))/hdisk/kpc_in_cm/kpc_in_cm/kpc_in_cm
			print 'cosmic ray energy density (eV/cm^3) around solar circle', creden_in_eV_per_cm3
			snaplist=np.append(snaplist,i)
			credenlist=np.append(credenlist,creden_in_eV_per_cm3)
                #plt.plot(snaplist,credenlist,label='kappa='+dclabel+'; '+'time step= '+timestep)
		#plt.plot(snaplist,credenlist,label='kappa='+dclabel)
		plt.plot(snaplist,credenlist,label=runtodo)
        plt.axhline(y=1.0,ls='--',color='k')
        plt.xlabel('Myr')
        plt.xlim(xmax=Nsnap)
        plt.title(runtitle+' CR around solar circle')
        plt.ylabel(r'$E_{\rm CR}({\rm eV/cm^3})$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/CR_solarcircle_'+fmeat+'.pdf')
        plt.clf()

if wanted=='crdenmidplane': #cosmic ray energy density at solar circle (observation ~ 1 eV/cm^3)
        for runtodo in dirneed:
                snaplist=[]
                credenlist=[]
		radlist=[]
		info=outdirname(runtodo, Nsnap)
		rundir=info['rundir']
		runtitle=info['runtitle']
		slabel=info['slabel']
		snlabel=info['snlabel']
		dclabel=info['dclabel']
		resolabel=info['resolabel']
		the_snapdir=info['the_snapdir']
		Nsnapstring=info['Nsnapstring']
		havecr=info['havecr']
		Fcal=info['Fcal']
		iavesfr=info['iavesfr']
		timestep=info['timestep']
		G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
		Gp = G['p']
		Grho = G['rho']
		Gu = G['u']
		Gm = G['m']
		cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
		Neb = G['ne']
		#Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
		Gz = Gp[:,2]
		Gx = Gp[:,0]
		Gy = Gp[:,1]
		dr = withinr/nogrid
		cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
		cregy_in_eV = cregy_in_erg*erg_in_eV
		for irad in range(nogrid):
			cutxy = (Gx*Gx+Gy*Gy > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy < dr*(irad+1)*dr*(irad+1))
			cutz = Gz*Gz < maxlength*maxlength
			cut = cutxy*cutz
			cregy_in_eV_cut = np.sum(cregy_in_eV[cut])
			shellvol_in_cm3 = np.pi*(-np.power(dr*irad,2)+np.power(dr*(irad+1),2))*kpc_in_cm*kpc_in_cm*kpc_in_cm*2.0*maxlength
			creden_in_eV_per_cm3 = cregy_in_eV_cut/shellvol_in_cm3
                        credenlist=np.append(credenlist,creden_in_eV_per_cm3)
			radlist=np.append(radlist,dr*irad)
                #plt.plot(radlist,credenlist,label='kappa='+dclabel+'; '+'time step= '+timestep)
                #plt.plot(radlist,credenlist,label='kappa='+dclabel)
		plt.plot(radlist,credenlist,label=runtodo)
        plt.axhline(y=1.0,ls='--',color='k')
        plt.xlabel('r (kpc)')
        plt.xlim(xmax=np.amax(radlist))
        plt.title(runtitle+' CR energy density at midplane')
        plt.ylabel(r'$E_{\rm CR}({\rm eV/cm^3})$')
        #plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/CR_midplane_'+fmeat+'.pdf')
        plt.clf()



if wanted=='crdenv': #cosmic ray energy density at solar circle (observation ~ 1 eV/cm^3)
        for runtodo in dirneed:
                snaplist=[]
                credenlist=[]
                info=outdirname(runtodo, Nsnap)
                rundir=info['rundir']
                runtitle=info['runtitle']
                slabel=info['slabel']
                snlabel=info['snlabel']
                dclabel=info['dclabel']
                resolabel=info['resolabel']
                the_snapdir=info['the_snapdir']
                Nsnapstring=info['Nsnapstring']
                havecr=info['havecr']
                Fcal=info['Fcal']
                iavesfr=info['iavesfr']
                timestep=info['timestep']
                G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                Gp = G['p']
                Grho = G['rho']
                Gu = G['u']
                Gm = G['m']
                cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                Neb = G['ne']
                #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                Gz = Gp[:,2]
                Gx = Gp[:,0]
                Gy = Gp[:,1]
                cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                cregy_in_eV = cregy_in_erg*erg_in_eV
                hdisk=5.
                rdisk = 3.
		dz = hdisk/nogrid
		zl=[]
                for iv in range(nogrid):
                        cutxy = (Gx*Gx+Gy*Gy < rdisk*rdisk)
                        cutz = (np.absolute(Gz) < dz*(iv+1.0)) & (np.absolute(Gz) > dz*(iv))
                        cut = cutxy*cutz
                        cregy_in_eV_cut = np.sum(cregy_in_eV[cut])
                        creden_in_eV_per_cm3 = cregy_in_eV_cut/(np.pi*rdisk*rdisk*dz*2.0)/kpc_in_cm/kpc_in_cm/kpc_in_cm
                        credenlist=np.append(credenlist,creden_in_eV_per_cm3)
			zl = np.append(zl,dz*(iv+0.5))
                plt.plot(zl,credenlist,label=runtodo)
        plt.axhline(y=1.0,ls='--',color='k')
        plt.xlabel('z (kpc)')
        plt.xlim(xmax=np.amax(zl))
        plt.title(' CR energy density within radius ='+str(rdisk)+' kpc')
        plt.ylabel(r'$E_{\rm CR}({\rm eV/cm^3})$')
        #plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/CR_vertical_'+fmeat+'.pdf')
        plt.clf()



if wanted=='crdenplanes': #cosmic ray energy density at solar circle (observation ~ 1 eV/cm^3)
        for runtodo in dirneed:
                snaplist=[]
                creden0list=[]
                creden1list=[]
                creden2list=[]
                info=outdirname(runtodo, Nsnap)
                rundir=info['rundir']
                runtitle=info['runtitle']
                slabel=info['slabel']
                snlabel=info['snlabel']
                dclabel=info['dclabel']
                resolabel=info['resolabel']
                the_snapdir=info['the_snapdir']
                Nsnapstring=info['Nsnapstring']
                havecr=info['havecr']
                Fcal=info['Fcal']
                iavesfr=info['iavesfr']
                timestep=info['timestep']
                G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                Gp = G['p']
                Grho = G['rho']
                Gu = G['u']
                Gm = G['m']
                cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                Neb = G['ne']
                #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                Gz = Gp[:,2]
                Gx = Gp[:,0]
                Gy = Gp[:,1]
                cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                cregy_in_eV = cregy_in_erg*erg_in_eV
                radlist = np.linspace(1.1,20,num=10)
                wdisk=2.
                hdisk = 0.25
                for radius in radlist:
                        insol = radius+wdisk/2.
                        outsol = radius-wdisk/2.
                        cutxy = (Gx*Gx+Gy*Gy < insol*insol) & (Gx*Gx+Gy*Gy > outsol*outsol)
                        cutz0 = np.absolute(Gz) < hdisk/2.
			cutz1 = np.absolute(Gz-1.0) < hdisk/2.
			cutz2 = np.absolute(Gz-2.0) < hdisk/2.
                        cut0 = cutxy*cutz0
			cut1 = cutxy*cutz1
			cut2 = cutxy*cutz2
                        cregy_in_eV_cut0 = np.sum(cregy_in_eV[cut0])
                        cregy_in_eV_cut1 = np.sum(cregy_in_eV[cut1])
                        cregy_in_eV_cut2 = np.sum(cregy_in_eV[cut2])
                        creden0_in_eV_per_cm3 = cregy_in_eV_cut0/(np.pi*(insol*insol-outsol*outsol))/hdisk/kpc_in_cm/kpc_in_cm/kpc_in_cm
                        creden1_in_eV_per_cm3 = cregy_in_eV_cut1/(np.pi*(insol*insol-outsol*outsol))/hdisk/kpc_in_cm/kpc_in_cm/kpc_in_cm
                        creden2_in_eV_per_cm3 = cregy_in_eV_cut2/(np.pi*(insol*insol-outsol*outsol))/hdisk/kpc_in_cm/kpc_in_cm/kpc_in_cm
                        creden0list=np.append(creden0list,creden0_in_eV_per_cm3)
			creden1list=np.append(creden1list,creden1_in_eV_per_cm3)
			creden2list=np.append(creden2list,creden2_in_eV_per_cm3)
                #plt.plot(radlist,credenlist,label='kappa='+dclabel+'; '+'time step= '+timestep)
                #plt.plot(radlist,credenlist,label='kappa='+dclabel)
                plt.plot(radlist,creden0list,label='midplane')
                plt.plot(radlist,creden1list,label='1kpc')
                plt.plot(radlist,creden2list,label='2kpc')
		plt.xlabel('r (kpc)')
		plt.xlim(xmax=np.amax(radlist))
		plt.title(runtitle+' CR energy density')
		plt.ylabel(r'$E_{\rm CR}({\rm eV/cm^3})$')
		plt.yscale('log')
		plt.legend(loc='best')
		plt.savefig('CRplot/CR_planes_'+runtodo+'.pdf')
		plt.clf()




if wanted=='cramap':
	rcParams['figure.figsize'] = 5, 5
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = 0
                for i in [Nsnap]:
			info=outdirname(runtodo, i)
			rundir=info['rundir'] 
			runtitle=info['runtitle']
			slabel=info['slabel']
			snlabel=info['snlabel']
			dclabel=info['dclabel']
			resolabel=info['resolabel']
			the_snapdir=info['the_snapdir']
			Nsnapstring=info['Nsnapstring']
			havecr=info['havecr']
			Fcal=info['Fcal']
			iavesfr=info['iavesfr']
			timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			Gpos = G['p']
			Gx = Gpos[:,0]
			Gy = Gpos[:,1]
			Gz = Gpos[:,2]
			Grho = G['rho']
			Gm = G['m']*1e10
                        cregyl = G['cregyl']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        cregyg = G['cregyg']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        cregy  = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        cregyd = G['cregyd']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        if havecr>4:
                                cregyp = G['cregyp']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        if havecr>5:
                                cregya = G['cregya']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
			print 'cregyp', cregyp
			print 'np.sum(cregyp)', np.sum(cregyp)
                        Hm, xedges, yedges = np.histogram2d(Gy, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregyl))
			plt.xlabel('x (kpc)')
			plt.ylabel('y (kpc)')
			im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
			plt.colorbar(im,fraction=0.046, pad=0.04)
			plt.tight_layout()
			plt.savefig('CRplot/'+runtodo+'_cregyl.pdf')
			plt.clf()
                        Hm, xedges, yedges = np.histogram2d(Gy, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregyg))
                        plt.xlabel('x (kpc)')
                        plt.ylabel('y (kpc)')
                        im = plt.imshow(np.log10(Hm),interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                        plt.colorbar(im,fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig('CRplot/'+runtodo+'_cregyg.pdf')
                        plt.clf()
                        Hm, xedges, yedges = np.histogram2d(Gy, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregyp))
                        plt.xlabel('x (kpc)')
                        plt.ylabel('y (kpc)')
                        im = plt.imshow(np.log10(Hm),interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                        plt.colorbar(im,fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig('CRplot/'+runtodo+'_cregyp.pdf')
                        plt.clf()
                        Hm, xedges, yedges = np.histogram2d(Gy, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregy))
                        plt.xlabel('x (kpc)')
                        plt.ylabel('y (kpc)')
                        im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                        plt.colorbar(im,fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig('CRplot/'+runtodo+'_cregy.pdf')
                        plt.clf()
                        Hm, xedges, yedges = np.histogram2d(Gy, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(Gm))
                        plt.xlabel('x (kpc)')
                        plt.ylabel('y (kpc)')
                        im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                        plt.colorbar(im,fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig('CRplot/'+runtodo+'_gm.pdf')
                        plt.clf()
			cread = cregy-cregyg-cregyd-cregyl
                        Hm, xedges, yedges = np.histogram2d(Gy, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cread))
                        plt.xlabel('x (kpc)')
                        plt.ylabel('y (kpc)')
                        im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                        plt.colorbar(im,fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig('CRplot/'+runtodo+'_cread.pdf')
                        plt.clf()

if wanted=='crarad':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = []
		crecuml = []
		crecumg = []
		crecuma = []
		crecumd = []
		crecum  = []
		crecump = []
                for i in [Nsnap]:
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
			Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']/1e6/3.2e7*2e53
                        cregyg = G['cregyg']/1e6/3.2e7*2e53
                        cregy  = G['cregy']/1e6/3.2e7*2e53
                        cregyd = G['cregyd']/1e6/3.2e7*2e53
                        if havecr>4:
                                cregyp = G['cregyp']/1e6/3.2e7*2e53
                        if havecr>5:
                                cregya = G['cregya']/1e6/3.2e7*2e53
			rad=np.linspace(0.1,withinr, num=nogrid)		
			for i in range(len(rad)):
				crecuml = np.append(crecuml,np.sum(cregyl[Gr<rad[i]]))
				crecumg = np.append(crecumg, np.sum(cregyg[Gr<rad[i]]))
				crecuma = np.append(crecuma, np.sum(cregya[Gr<rad[i]]))
				crecum = np.append(crecum, np.sum(cregy[Gr<rad[i]]))
				crecumd = np.append(crecumd, np.sum(cregyd[Gr<rad[i]]))
				crecump = np.append(crecump, np.sum(cregyp[Gr<rad[i]]))
			crecumad = crecumd + crecumg
			plt.plot(rad, crecum,label='CR energy')
			plt.plot(rad, crecumg, label='SNe')
			plt.plot(rad, crecuml, label='Loss')
			plt.plot(rad, crecumd-crecuml-crecump, label='Ad')
			plt.plot(rad, crecuma, label='test Ad')
			plt.plot(rad, crecump, label='Flux')
			plt.plot(rad, crecumad, label='SNe+Loss+Ad+Flux')
			plt.legend()
			plt.ylabel(r'$\left \langle \rm{dE/dt}  \right \rangle$ (erg/s)')
			plt.xlabel('r (kpc)')
			plt.savefig('CRplot/'+runtodo+'_rad.pdf')
			plt.clf()

if wanted=='sfrrad':
        for runtodo in dirneed:
		info=outdirname(runtodo, Nsnap)
		rundir=info['rundir']
		runtitle=info['runtitle']
		slabel=info['slabel']
		snlabel=info['snlabel']
		dclabel=info['dclabel']
		resolabel=info['resolabel']
		the_snapdir=info['the_snapdir']
		Nsnapstring=info['Nsnapstring']
		havecr=info['havecr']
		Fcal=info['Fcal']
		iavesfr=info['iavesfr']
		timestep=info['timestep']
                S = readsnapcr(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
                Sage = S['age']
		Spos = S['p']
		Sx = Spos[:,0]
		Sy = Spos[:,1]
		Sz = Spos[:,2]
                Sm = S['m']
		#print 'Sage', Sage, np.amax(Sage), np.amin(Sage)
		print 'Sage', Sage
                radlist = np.linspace(0.5,withinr,num=10)
                wdisk=radlist[1]-radlist[0]
                hdisk = maxlength*2.
		agecutlow = 0.20 #in Gyr
		agecuthigh = 0.25
		Smdenlist = []
                for radius in radlist:
                        insol = radius+wdisk/2.
                        outsol = radius-wdisk/2.
                        cutxy = (Sx*Sx+Sy*Sy < insol*insol) & (Sx*Sx+Sy*Sy > outsol*outsol)
                        cutz = np.absolute(Sz) < hdisk/2.
			cuta = (Sage> agecutlow) & (Sage <agecuthigh)
                        cut = cutxy*cutz*cuta
			Smdencut = np.sum(Sm[cut])*1e10/(np.pi*(insol*insol-outsol*outsol))/hdisk #in Msun/kpc^3
                        Smdenlist=np.append(Smdenlist,Smdencut)
		Smdenlist = np.array(Smdenlist)
		sfrden = Smdenlist/(agecuthigh-agecutlow)/1.0e9
                #plt.plot(radlist,sfrden,label='kappa='+dclabel+'; '+'time step= '+timestep)
		plt.plot(radlist,sfrden,label=runtodo)
		print 'sfrden', sfrden
        plt.xlabel('r (kpc)')
        plt.xlim(xmax=np.amax(radlist))
        plt.title(runtitle+' SFR density at midplane')
        plt.ylabel(r'$\rho_{\rm SFR}({\rm M_{\odot}/yr/kpc^3})$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/SFR_midplane_'+fmeat+'.pdf')


if wanted=='sfrv':
        for runtodo in dirneed:
                info=outdirname(runtodo, Nsnap)
                rundir=info['rundir']
                runtitle=info['runtitle']
                slabel=info['slabel']
                snlabel=info['snlabel']
                dclabel=info['dclabel']
                resolabel=info['resolabel']
                the_snapdir=info['the_snapdir']
                Nsnapstring=info['Nsnapstring']
                havecr=info['havecr']
                Fcal=info['Fcal']
                iavesfr=info['iavesfr']
                timestep=info['timestep']
                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
                Sage = S['age']
                Spos = S['p']
                Sx = Spos[:,0]
                Sy = Spos[:,1]
                Sz = Spos[:,2]
                Sm = S['m']
                #print 'Sage', Sage, np.amax(Sage), np.amin(Sage)
                rdisk=3.
                height=2.
		dz=height/nogrid
                agecut = 0.2 #in Gyr
                Smdenlist = []
		zl =[]
		for iv in range(nogrid):
			cutxy = (Sx*Sx+Sy*Sy < rdisk*rdisk)
			cutz = (Sz*Sz<(iv+1.0)*dz*(iv+1.0)*dz)&(Sz*Sz > iv*dz*iv*dz)
                        cuta = Sage< agecut
                        cut = cutxy*cutz*cuta
			vol_in_kpc3 = np.pi*(rdisk*rdisk*dz*2)
                        Smdencut = np.sum(Sm[cut])*1e10/vol_in_kpc3 #in Msun/kpc^3
                        Smdenlist=np.append(Smdenlist,Smdencut)
			zl = np.append(zl,(iv+0.5)*dz)
                Smdenlist = np.array(Smdenlist)
                sfrden = Smdenlist/agecut/1.0e9
                plt.plot(zl,sfrden,label=runtodo)
                print 'sfrden', sfrden
        plt.xlabel('z (kpc)')
        plt.xlim(xmax=np.amax(zl))
        plt.title(runtitle+' SFR density at within = '+str(rdisk)+'kpc')
        plt.ylabel(r'$\rho_{\rm SFR}({\rm M_{\odot}/yr/kpc^3})$')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/SFR_vertical_'+fmeat+'.pdf')


if wanted=='sfrarad':
        for runtodo in dirneed:
                rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, Nsnap)
                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix)
                Sage = S['age']
                Spos = S['p']
                Sx = Spos[:,0]
                Sy = Spos[:,1]
                Sz = Spos[:,2]
                Sm = S['m']
                #print 'Sage', Sage, np.amax(Sage), np.amin(Sage)
                radlist = np.linspace(1.1,20,num=10)
                wdisk=2.
                hdisk = 1.
                agecut = 0.2 #in Gyr
                Smlist = []
                for radius in radlist:
                        insol = radius+wdisk/2.
                        outsol = radius-wdisk/2.
                        cutxy = (Sx*Sx+Sy*Sy < insol*insol)
                        cutz = np.absolute(Sz) < hdisk/2.
                        cuta = Sage< agecut
                        cut = cutxy*cutz*cuta
                        Smcut = np.sum(Sm[cut])*1e10 #in Msun
                        Smlist=np.append(Smlist,Smcut)
                Smlist = np.array(Smlist)
                sfrcum = Smlist/agecut/1.0e9 #in Msun/yr
                #plt.plot(radlist,sfrden,label='kappa='+dclabel+'; '+'time step= '+timestep)
                plt.plot(radlist,sfrcum,label=runtodo)
                print 'sfrden', sfrcum
        plt.xlabel('r (kpc)')
        plt.xlim(xmax=np.amax(radlist))
        plt.title(runtitle+' cumulative SFR at midplane')
        plt.ylabel(r'$M_{\rm *,new}({\rm M_{\odot}/yr})$')
        #plt.yscale('log')
        plt.legend(loc='best')
        plt.savefig('CRplot/SFR_cum_midplane_'+fmeat+'.pdf')
        plt.clf()


if wanted=='crcpsnap':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = 0
                for i in range(startno,Nsnap, snapsep):
                        rundir, runtitle, slabel, snlabel, dclabel, resolabel, the_snapdir, Nsnapstring, havecr, Fcal, iavesfr, timestep=outdirname(runtodo, i)
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        cregyl = G['cregyl']
                        cregyg = G['cregyg']
                        cregy  = G['cregy']
                        cregyd = G['cregyd']
                        if havecr>4:
                                cregyp = G['cregyp']
                        try:
                                Sm = np.sum(S['m'])
                        except KeyError:
                                Sm = 0.
                        cregygt=np.sum(cregyg)
                        cregyt =np.sum(cregy)
                        cregylt=np.sum(cregyl)
                        cregydt=np.sum(cregyd)
                        if havecr>4:
                                cregydtp=np.sum(cregyp)
                        eng = (cregygt-preg)/1e6/3.2e7*2e53/float(snapsep)
                        enc = (cregyt-prec)/1e6/3.2e7*2e53/float(snapsep)
                        enl = (cregylt-prel)*2e53/1e6/3.2e7/float(snapsep)
                        end = (cregydt-pred)*2e53/1e6/3.2e7/float(snapsep)
                        if havecr>4:
                                enp = (cregydtp-prep)*2e53/1e6/3.2e7/float(snapsep)
                                print 'CR energy dtp', enp
                        print 'CR energy loss rate (erg/s)', enl
                        print 'CR energy gain rate (erg/s)', eng
                        print 'CR energy change rate (erg/s)', enc
                        print 'CR energy dt rate (erg/s)', end  #including adiabatic heating and streaming
                        snaplist.append(i)
                        enclist.append(enc)
                        englist.append(eng)
                        enllist.append(enl)
                        endlist.append(end)
                        if havecr>4:
                                enplist.append(enp)
                        preg = cregygt
                        prec = cregyt
                        prel = cregylt
                        pred = cregydt
                        if havecr>4:
                                prep = cregydtp
                        if i>presnap:
                                avesfr=(Sm-presm)*1e4/0.98/(i-presnap)
                        else:
                                avesfr=0
                        presnap = i
                        presm = Sm
                        print 'avesfr', avesfr
                        avesfrl.append(avesfr)
                avesfrl=np.array(avesfrl)
                enclist=np.array(enclist)
                englist=np.array(englist)
                endlist=np.array(endlist)
                enllist=np.array(enllist)
                enplist=np.array(enplist)
                plt.plot(snaplist[1:], enclist[1:], label='Change')
                plt.plot(snaplist[1:], englist[1:], label='SNe')
                if havecr > 4:
                        plt.plot(snaplist[1:], endlist[1:], label='Ad')
                #plt.plot(snaplist, endlist, label='CR energy dt')
                plt.plot(snaplist[1:], enllist[1:], label='Cool')
                #plt.plot(snaplist, englist+endlist+enllist, label='CR energy estimate')
                if havecr>4:
                        plt.plot(snaplist[1:], englist[1:]+endlist[1:]+enllist[1:], label='SNe+Ad+Cool')
                #plt.yscale('log')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel('Myr', fontsize=25)
                plt.subplots_adjust(left=0.2,bottom=0.2, right=0.75)
                plt.ylabel('dE/dt (erg/s)', fontsize=25)
                plt.savefig('CRplot/cresnap_'+fmeat+'.pdf')
                plt.clf()
                #Lsfr = 3.8e-4 *avesfr*2e33/3.2e7*9e20
                #above is the coefficient for Salpeter only; for Kroupa, the coefficient is 50% larger:
                Lsfr = 3.8e-4 *1.5*avesfrl*2e33/3.2e7*9e20
                Lsfrideal = 3.8e-4 * 1.5 * 1.0 * 2e33 / 3.2e7 *9e20
                if havecr >4:
                        Lgammae = (enllist+endlist)/7.51e-16/2e5/3.2e7/250.0*0.7/3.0
                Lgamma = (enllist)/7.51e-16/2e5/3.2e7/250.0*0.7/3.0
                #Lgammae_sfr = Lgammae/Lsfr
                Lgamma_sfr = Lgamma/Lsfr
                plt.plot(snaplist[1:], np.absolute(Lgamma_sfr[1:]), label=r'Cool')
                #plt.plot(snaplist[1:], np.absolute(Lgammae_sfr[1:]), label='Cool+ad')
                #plt.plot(snaplist[1:], np.absolute(Lgamma[1:]/Lsfrideal), label='SFR=1')
                plt.yscale('log')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel('Myr', fontsize=25)
                plt.ylabel(r'$\frac{L_{\gamma}}{L_{\rm SF}}$', fontsize=30)
                plt.savefig('CRplot/gammasfrsnap_'+fmeat+'.pdf')
                plt.clf()

if wanted=='crdrad':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = []
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']/1e6/3.2e7*2e53
                        cregyg = G['cregyg']/1e6/3.2e7*2e53
                        cregy  = G['cregy']/1e6/3.2e7*2e53
                        cregyd = G['cregyd']/1e6/3.2e7*2e53
                        if havecr>4:
                                cregyp = G['cregyp']/1e6/3.2e7*2e53
                        if havecr>5:
                                cregya = G['cregya']/1e6/3.2e7*2e53
                        rad=np.linspace(0.1,withinr, num=nogrid)
                        for i in range(len(rad)-1):
				cutu=Gr<rad[i+1]
				cutd=Gr>rad[i]
				cutz=np.absolute(Gz)<maxlength
				cut=cutu*cutd*cutz
				vol = 2.0*maxlength*np.pi*(np.power(rad[i+1],2)-np.power(rad[i],2))
                                crecuml = np.append(crecuml,np.sum(cregyl[cut])/vol)
                                crecumg = np.append(crecumg, np.sum(cregyg[cut])/vol)
                                crecuma = np.append(crecuma, np.sum(cregya[cut])/vol)
                                crecum = np.append(crecum, np.sum(cregy[cut])/vol)
                                crecumd = np.append(crecumd, np.sum(cregyd[cut])/vol)
                                crecump = np.append(crecump, np.sum(cregyp[cut])/vol)
                        crecumad = crecumd + crecumg
                        plt.plot(rad[1:], crecum,label='CR energy')
                        plt.plot(rad[1:], crecumg, label='SNe')
                        plt.plot(rad[1:], crecuml, label='Loss')
                        plt.plot(rad[1:], crecumd-crecuml-crecump, label='Ad')
                        plt.plot(rad[1:], crecuma, label='test Ad')
                        plt.plot(rad[1:], crecump, label='Flux')
                        plt.plot(rad[1:], crecumad, label='SNe+Loss+Ad+Flux')
                        plt.legend()
			plt.title('CR density within the disk')
                        plt.ylabel(r'$\left \langle \rm{de/dt}  \right \rangle {\rm (erg/s/kpc^3)}$')
                        plt.xlabel('r (kpc)')
                        plt.savefig('CRplot/'+runtodo+'_crdrad.pdf')
                        plt.clf()


if wanted=='cratime':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
		prea=0
                presm = 0
                presnap = []
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
		timel = []
                for i in range(startno,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
			cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
			if cosmo==1:
				G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr, cosmological=1,h0=1)
			else:
                        	G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']*2e53
                        cregyg = G['cregyg']*2e53
                        cregy  = G['cregy']*2e53
                        cregyd = G['cregyd']*2e53
                        if havecr>4:
                                cregyp = G['cregyp']*2e53
                        if havecr>5:
                                cregya = G['cregya']*2e53
                        crecum = np.append(crecum, np.sum(cregy))
			if fmeat == '573':
                                crecuml = np.append(crecuml,np.sum(cregyl)+prel)
                                crecumg = np.append(crecumg, np.sum(cregyg)+preg)
                                crecuma = np.append(crecuma, np.sum(cregya)+prea)
                                crecumd = np.append(crecumd, np.sum(cregyd)+pred)
                                crecump = np.append(crecump, np.sum(cregyp)+prep)
			else:
				crecuml = np.append(crecuml,np.sum(cregyl))
				crecumg = np.append(crecumg, np.sum(cregyg))
				crecuma = np.append(crecuma, np.sum(cregya))
				crecumd = np.append(crecumd, np.sum(cregyd))
				crecump = np.append(crecump, np.sum(cregyp))
			print 'cregy', np.sum(cregy)
			print 'cregyp', np.sum(cregyp)
			if cosmo==1:
				snap2list, time2list=readtime(firever=2)
				timenow = np.interp(i,snap2list,time2list)*1e3		
			else:
				timenow=i*0.98
			timel = np.append(timel, timenow)
			prec=np.sum(cregy)
			prel=np.sum(cregyl)
			preg=np.sum(cregyg)
			prea=np.sum(cregya)
			pred=np.sum(cregyd)
			prep=np.sum(cregyp)
		if useM1==1:
			crecumad = crecumd + crecumg + crecuml
		else:
			crecumad = crecumd + crecumg
		plt.plot(timel, crecum, marker='s',label='CR energy')
		plt.plot(timel, crecumg, marker='s',label='SNe')
		plt.plot(timel, crecuml, marker='s',label='Loss')
#		plt.plot(timel, crecumd-crecuml-crecump, label='Ad')
		plt.plot(timel, crecuma,marker='s', label='Ad')
		plt.plot(timel, crecump, marker='s',label='Flux')
		plt.plot(timel, crecumad, marker='s',label='Sum')
		plt.legend(bbox_to_anchor=(1.1, 1.05))
		plt.ylabel(r'$\left \langle \rm{E}  \right \rangle$ (erg)')
		plt.xlabel('t (Myr)')
		#plt.ylim([-3.0e43,4.0e43])
		plt.title('diffusion coefficient = ' +dclabel) 
		plt.savefig('CRplot/'+runtodo+'_time.pdf')
		plt.clf()

if wanted=='crdtime':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
		prea=0
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = []
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
                timel = []
                for i in range(0,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']/1e6/3.2e7*2e53
                        cregyg = G['cregyg']/1e6/3.2e7*2e53
                        cregy  = G['cregy']/1e6/3.2e7*2e53
                        cregyd = G['cregyd']/1e6/3.2e7*2e53
                        if havecr>4:
                                cregyp = G['cregyp']/1e6/3.2e7*2e53
                        if havecr>5:
                                cregya = G['cregya']/1e6/3.2e7*2e53
                        crecuml = np.append(crecuml,(np.sum(cregyl)-prel)/(i*0.98*snapsep))
                        crecumg = np.append(crecumg, (np.sum(cregyg)-preg)/(i*0.98*snapsep))
                        crecuma = np.append(crecuma, (np.sum(cregya)-prea)/(i*0.98*snapsep))
                        crecum = np.append(crecum, (np.sum(cregy)-prec)/(i*0.98*snapsep))
                        crecumd = np.append(crecumd, (np.sum(cregyd)-pred)/(i*0.98*snapsep))
                        crecump = np.append(crecump, (np.sum(cregyp)-prep)/(i*0.98*snapsep))
                        timel = np.append(timel, i*0.98)
			prel=np.sum(cregyl)
			preg=np.sum(cregyg)
			prea=np.sum(cregya)
			prec=np.sum(cregy)
			pred=np.sum(cregyd)
			prep=np.sum(cregyp)
                crecumad = crecumd + crecumg
                plt.plot(timel, crecum,label='CR energy')
                plt.plot(timel, crecumg, label='SNe')
                plt.plot(timel, crecuml, label='Loss')
                plt.plot(timel, crecumd-crecuml-crecump, label='Ad')
                #plt.plot(rad, crecuma, label='test Ad')
                plt.plot(timel, crecump, label='Flux')
                plt.plot(timel, crecumad, label='SNe+Loss+Ad+Flux')
                plt.legend()
                plt.ylabel(r'$\left \langle \rm{dE/dt}  \right \rangle$ (erg/s)')
                plt.xlabel('t (Myr)')
                plt.savefig('CRplot/'+runtodo+'_dtime.pdf')


if wanted=='gasdenmidplane':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
			print 'Gm', Gm
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        dr = withinr/nogrid
                        Gnism_in_cm_3l=[]
                        radl =[]
                        for irad in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy < dr*(irad+1)*dr*(irad+1))
				cutz = Gz*Gz < maxlength*maxlength
				cut=cutxy*cutz
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gm_in_g = Gmcut*1e10*solar_mass_in_g
                                shellvol_in_cm3 = np.pi*(-np.power(dr*irad,2)+np.power(dr*(irad+1),2))*3.086e21*3.086e21*3.086e21*2.0*maxlength
                                Grho_in_g_cm_3 = np.sum(Gm_in_g)/shellvol_in_cm3
				try:
					Nebave = np.average(Nebcut,weights=Gmcut)
				except ZeroDivisionError:
					Nebave = 0
                                Gnism_in_cm_3 = (0.78+0.22*Nebave*0.76)/proton_mass_in_g*Grho_in_g_cm_3
                                Gnism_in_cm_3l = np.append(Gnism_in_cm_3l, Gnism_in_cm_3)
                                radl = np.append(radl, dr*(irad+0.5))
                        plt.plot(radl, Gnism_in_cm_3l, label=runtodo)
        plt.xlabel('r [kpc]')
        plt.ylabel(r'$n_{\rm ISM} [{\rm cm^{-3}}]$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('midplane gas density')
        plt.savefig('CRplot/gasdensity_'+fmeat+'_midplane.pdf')
        plt.clf()


if wanted=='gasdenv':
        for runtodo in dirneed:
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        #Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
			withinz=5.0
			rdisk=3.0
                        dz = withinz/nogrid
                        Gnism_in_cm_3l=[]
                        zl =[]
                        for iv in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy < rdisk*rdisk)
                                cutz = (Gz*Gz<(iv+1.0)*dz*(iv+1.0)*dz)&(Gz*Gz > iv*dz*iv*dz)
                                cut=cutxy*cutz
                                Nebcut = Neb[cut]
                                Gmcut = Gm[cut]
                                Gm_in_g = Gmcut*1e10*2e33
                                vol_in_cm3 = np.pi*(rdisk*rdisk*3.086e21*3.086e21*3.086e21*dz*2)
                                Grho_in_g_cm_3 = Gm_in_g/vol_in_cm3
                                protonmass_in_g = 1.67e-24
                                Gnism_in_cm_3 = np.sum((0.78+0.22*Nebcut*0.76)/protonmass_in_g*Grho_in_g_cm_3)
                                Gnism_in_cm_3l = np.append(Gnism_in_cm_3l, Gnism_in_cm_3)
                                zl = np.append(zl, dz*(iv+0.5))
                        plt.plot(zl, Gnism_in_cm_3l, label=runtodo)
        plt.xlabel('z [kpc]')
        plt.ylabel(r'$n_{\rm ISM} [{\rm cm^{-3}}]$')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.title('vertical gas density within radius = '+str(rdisk)+'kpc')
        plt.savefig('CRplot/gasdensity_'+fmeat+'_v.pdf')
        plt.clf()



if wanted=='lgratio':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = []
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
                timel = []
                for i in range(0,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']/1e6/3.2e7*2e53
                        cregyg = G['cregyg']/1e6/3.2e7*2e53
                        cregy  = G['cregy']/1e6/3.2e7*2e53
                        cregyd = G['cregyd']/1e6/3.2e7*2e53
                        if havecr>4:
                                cregyp = G['cregyp']/1e6/3.2e7*2e53
                        if havecr>5:
                                cregya = G['cregya']/1e6/3.2e7*2e53
                        crecuml = np.append(crecuml,np.sum(cregyl))
                        crecumg = np.append(crecumg, np.sum(cregyg))
                        crecuma = np.append(crecuma, np.sum(cregya))
                        crecum = np.append(crecum, np.sum(cregy))
                        crecumd = np.append(crecumd, np.sum(cregyd))
                        crecump = np.append(crecump, np.sum(cregyp))
                        timel = np.append(timel, i*0.98)
		crecumg=np.array(crecumg)
		crecuml=np.array(crecuml)
                plt.plot(timel, crecuml/crecumg, label=runtodo)
                plt.legend(bbox_to_anchor=(1.1, 1.05))
                plt.ylabel(r'$\left \langle \rm{E_{\bf Loss}/E_{\bf SNe}}  \right \rangle$')
                plt.xlabel('t (Myr)')
                plt.title('diffusion coefficient = ' +dclabel)
	plt.savefig('CRplot/lgratio_'+fmeat+'.pdf')


if wanted=='dlgratio':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = []
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
                timel = []
                for i in range(0,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']/1e6/3.2e7*2e53
                        cregyg = G['cregyg']/1e6/3.2e7*2e53
                        cregy  = G['cregy']/1e6/3.2e7*2e53
                        cregyd = G['cregyd']/1e6/3.2e7*2e53
                        if havecr>4:
                                cregyp = G['cregyp']/1e6/3.2e7*2e53
                        if havecr>5:
                                cregya = G['cregya']/1e6/3.2e7*2e53
                        crecuml = np.append(crecuml,np.sum(cregyl))
                        crecumg = np.append(crecumg, np.sum(cregyg))
                        crecuma = np.append(crecuma, np.sum(cregya))
                        crecum = np.append(crecum, np.sum(cregy))
                        crecumd = np.append(crecumd, np.sum(cregyd))
                        crecump = np.append(crecump, np.sum(cregyp))
                        timel = np.append(timel, i*0.98)
                crecumg=np.array(crecumg)
                crecuml=np.array(crecuml)
		plt.ylim([0.01,1.5])
                plt.plot(timel[1:], (-crecuml[1:]+crecuml[:-1])/(crecumg[1:]-crecumg[:-1]), label=runtodo)
                plt.legend(loc='best',fontsize=16)
		plt.yscale('log')
                plt.ylabel(r'$\left \langle \rm{\Delta E_{Loss}/\Delta E_{SNe}}  \right \rangle$')
                plt.xlabel('t (Myr)')
                plt.title('diffusion coefficient = ' +dclabel)
        plt.savefig('CRplot/dlgratio_'+fmeat+'.pdf')

if wanted=='gsmratio':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = []
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
                timel = []
		smen = []
                for i in range(startno,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
			SMtot = np.sum(S['m'])*1e10
			Energysm = SMtot*1.989e30*3.0e8*3.0e8/1e6/3.2e7*1e7*6.2e-4
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']/1e6/3.2e7*2e53
                        cregyg = G['cregyg']/1e6/3.2e7*2e53
                        cregy  = G['cregy']/1e6/3.2e7*2e53
                        cregyd = G['cregyd']/1e6/3.2e7*2e53
                        if havecr>4:
                                cregyp = G['cregyp']/1e6/3.2e7*2e53
                        if havecr>5:
                                cregya = G['cregya']/1e6/3.2e7*2e53
                        crecuml = np.append(crecuml,np.sum(cregyl))
                        crecumg = np.append(crecumg, np.sum(cregyg))
                        crecuma = np.append(crecuma, np.sum(cregya))
                        crecum = np.append(crecum, np.sum(cregy))
                        crecumd = np.append(crecumd, np.sum(cregyd))
                        crecump = np.append(crecump, np.sum(cregyp))
			smen = np.append(smen, Energysm)
                        timel = np.append(timel, i*0.98)
                crecumg=np.array(crecumg)
                smen=np.array(smen)
                plt.plot(timel, crecumg/smen, label=runtodo)
                plt.legend(bbox_to_anchor=(1.1, 1.05))
                plt.ylabel(r'$\left \langle \rm{E_{SNe}/E_{SFR_to_SNe}}  \right \rangle$')
                plt.xlabel('t (Myr)')
                plt.title('diffusion coefficient = ' +dclabel)
        plt.savefig('CRplot/gsmratio_'+fmeat+'.pdf')


if wanted=='cramapv':
        rcParams['figure.figsize'] = 5, 5
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                avesfrl=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = 0
                for i in [Nsnap]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10 #in solar mass
			if havecr>0:
				cregyl = G['cregyl']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
				cregyg = G['cregyg']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
				cregy  = G['cregy']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
				cregyd = G['cregyd']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        if havecr>4:
                                cregyp = G['cregyp']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
                        if havecr>5:
                                cregya = G['cregya']*1e10*solar_mass_in_g*km_in_cm*km_in_cm
			if havecr>0:
				Hm, xedges, yedges = np.histogram2d(Gz, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregyl))
				plt.xlabel('x (kpc)')
				plt.ylabel('z (kpc)')
				im = plt.imshow(np.log10(Hm),  interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
				plt.colorbar(im,fraction=0.046, pad=0.04)
				plt.tight_layout()
				plt.savefig('CRplot/'+runtodo+'_cregyl_v.pdf')
				plt.clf()
				Hm, xedges, yedges = np.histogram2d(Gz, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregyg))
				plt.xlabel('x (kpc)')
				plt.ylabel('z (kpc)')
				im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
				plt.colorbar(im,fraction=0.046, pad=0.04)
				plt.tight_layout()
				plt.savefig('CRplot/'+runtodo+'_cregyg_v.pdf')
				plt.clf()
                                Hm, xedges, yedges = np.histogram2d(Gz, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregy))
                                plt.xlabel('x (kpc)')
                                plt.ylabel('z (kpc)')
                                im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])  
                                plt.colorbar(im,fraction=0.046, pad=0.04)
                                plt.tight_layout()
                                plt.savefig('CRplot/'+runtodo+'_cregy_v.pdf')
                                plt.clf()
			if havecr>4:
				Hm, xedges, yedges = np.histogram2d(Gz, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cregyp))
				plt.xlabel('x (kpc)')
				plt.ylabel('z (kpc)')
				im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
				plt.colorbar(im,fraction=0.046, pad=0.04)
				plt.tight_layout()
				plt.savefig('CRplot/'+runtodo+'_cregyp_v.pdf')
				plt.clf()
			if havecr>5:
				cread = cregy-cregyg-cregyd-cregyl
				Hm, xedges, yedges = np.histogram2d(Gz, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(cread))
				plt.xlabel('x (kpc)')
				plt.ylabel('z (kpc)')
				im = plt.imshow(np.log10(Hm), interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
				plt.colorbar(im,fraction=0.046, pad=0.04)
				plt.tight_layout()
				plt.savefig('CRplot/'+runtodo+'_cread_v.pdf')
				plt.clf()
                        Hm, xedges, yedges = np.histogram2d(Gz, Gx, bins=100,range=[[-withinr,withinr],[-withinr,withinr]], weights=np.absolute(Gm))
                        plt.xlabel('x (kpc)')
                        plt.ylabel('z (kpc)')
                        im = plt.imshow(np.log10(Hm),  interpolation='nearest', origin='low',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])   
                        plt.colorbar(im,fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        plt.savefig('CRplot/'+runtodo+'_gm_v.pdf')
                        plt.clf()




if wanted=='dirage':
        for runtodo in dirneed:
                snaplist=[]
                avesfrl=[]
                presm = 0
                presnap = 0
                for i in [startno]:
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        if cosmo==1:
                                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
                        else:
                                S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
			print 'Sage', S['age']
			print 'mass ave age', np.sum(S['age']*S['m'])/np.sum(S['m'])



if wanted=='dg_sfr':
        for runtodo in dirneed:
                snaplist=[]
                enclist=[]
                englist=[]
                enllist=[]
                endlist=[]
                enplist=[]
                enalist=[]
                sml=[]
		nsml=[]
                prel=0
                preg=0
                prec=0
                pred=0
                prep=0
                presm = 0
                presnap = 0
		pretime = 0
                crecuml = []
                crecumg = []
                crecuma = []
                crecumd = []
                crecum  = []
                crecump = []
                timel = []
                for i in range(0,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
			cosmo=info['cosmo']
                        print 'the_snapdir', the_snapdir
                        print 'Nsnapstring', Nsnapstring
                        print 'havecr', havecr
                        if cosmo==1:
                                S = readsnapcr(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
                        else:
                                S = readsnapcr(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        try:
                                Sm = np.sum(S['m'])
                                Smi = S['m']
                                Sage = S['age']
                                Sm = np.sum(Smi)
                                header=S['header']
                                timeneed=header[2]
                                tcut=Sage>timeneed-0.001
                                nsm = np.sum(Smi[tcut])
                        except KeyError:
                                Sm = 0.
				timeneed=0.
				nsm=0.
                        snaplist.append(i)
			if cosmo==1:
				G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr,h0=1,cosmological=1)
			else:
				G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gpos = G['p']
                        Gx = Gpos[:,0]
                        Gy = Gpos[:,1]
                        Gz = Gpos[:,2]
                        Grho = G['rho']
                        Gm = G['m']*1e10
                        Gr = np.sqrt(np.square(Gx)+np.square(Gy)+np.square(Gz))
                        cregyl = G['cregyl']*2e53
                        cregyg = G['cregyg']*2e53
                        cregy  = G['cregy']*2e53
                        cregyd = G['cregyd']*2e53
                        if havecr>4:
                                cregyp = G['cregyp']*2e53
                        if havecr>5:
                                cregya = G['cregya']*2e53
                        crecuml = np.append(crecuml,np.sum(cregyl))
                        crecumg = np.append(crecumg, np.sum(cregyg))
                        crecuma = np.append(crecuma, np.sum(cregya))
                        crecum = np.append(crecum, np.sum(cregy))
                        crecumd = np.append(crecumd, np.sum(cregyd))
                        crecump = np.append(crecump, np.sum(cregyp))
                        timel = np.append(timel, i*0.98)
			sml=np.append(sml,Sm)
			nsml=np.append(nsml,nsm)
			pretime=timeneed
                crecumg=np.array(crecumg)
                crecuml=np.array(crecuml)
		prefac = 1e10/0.4*0.0037*1.0e51
		snesfr = sml*1e10/0.4*0.0037*1.0e51
		dg_sfr=(crecumg[1:]-crecumg[:-1])/(nsml[1:]*prefac)/snapsep
		print 'dg_sfr', dg_sfr
                plt.plot(timel[1:], dg_sfr, label=runtodo)
		plt.yscale('log')
                plt.legend(loc='best')
                plt.ylabel(r'$\left \langle \rm{\Delta E_{SNe,CR}/\Delta E_{SNe,SFR}}  \right \rangle$')
                plt.xlabel('t (Myr)')
                plt.title('diffusion coefficient = ' +dclabel)
        plt.savefig('CRplot/dg_sfr_'+fmeat+'.pdf')


if wanted=='gamma_partit':
        for runtodo in dirneed:
		Lgammal=[]
		Lgammapl=[]
		enllist=[]
		timel=[]
		Gmcutwl=[]
		cregywl=[]
		Nebcutal=[]
                for i in range(startno,Nsnap,snapsep):
                        info=outdirname(runtodo, i)
                        rundir=info['rundir']
                        runtitle=info['runtitle']
                        slabel=info['slabel']
                        snlabel=info['snlabel']
                        dclabel=info['dclabel']
                        resolabel=info['resolabel']
                        the_snapdir=info['the_snapdir']
                        Nsnapstring=info['Nsnapstring']
                        havecr=info['havecr']
                        Fcal=info['Fcal']
                        iavesfr=info['iavesfr']
                        timestep=info['timestep']
                        cosmo=info['cosmo']
                        G = readsnapcr(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, havecr=havecr)
                        Gp = G['p']
                        Grho = G['rho']
                        Gu = G['u']
                        Gm = G['m']
                        cregy = G['cregy'] #cosmic ray energy in 1e10Msun km^2/sec^2
                        Neb = G['ne']
			cregyl = G['cregyl']*1e10*solar_mass_in_g*km_in_cm*km_in_cm #in erg
                        #Gnism = (0.78+0.22*Neb*0.76)/1.67e-24*Grho*1e10*1.99e33/3.086e21/3.086e21/3.086e21 #gas number density in ISM 
                        Gz = Gp[:,2]
                        Gx = Gp[:,0]
                        Gy = Gp[:,1]
                        Gnism_in_cm_3 = (0.78+0.22*Neb*0.76)/proton_mass_in_g*Grho*1e10*solar_mass_in_g/kpc_in_cm/kpc_in_cm/kpc_in_cm
                        tpi_in_s = pidecay_fac/Gnism_in_cm_3 #pi decay time in s
                        cregy_in_erg = cregy*solar_mass_in_g*1e10*km_in_cm*km_in_cm
                        Lgammagev = cregy_in_erg/tpi_in_s*betapi/nopi_per_gamma #in erg/s
			cutxy = Gx*Gx+Gy*Gy < withinr*withinr
			cutz = np.absolute(Gz)<maxlength
			cut = cutxy*cutz
			Lgammacut = Lgammagev[cut]
			Gmcutw = np.sum(Gm[cut])
			cregyw = np.sum(cregy[cut])
			try:
				Nebcuta = np.average(Neb[cut],weights=Gm[cut])
			except ZeroDivisionError:
				Nebcuta = 0
                        dr = withinr/nogrid
			Lgammap=0.0	
                        for irad in range(nogrid):
                                cutxy = (Gx*Gx+Gy*Gy > dr*irad*dr*irad) & (Gx*Gx+Gy*Gy < dr*(irad+1)*dr*(irad+1))
                                cutz = Gz*Gz < maxlength*maxlength
                                cutr=cutxy*cutz
                                Nebcut = Neb[cutr]
                                Gmcut = Gm[cutr]
                                try:
                                        Nebave = np.average(Nebcut,weights=Gmcut)
                                except ZeroDivisionError:
                                        Nebave = 0
                                Gm_in_g = Gmcut*1e10*2e33
                                shellvol_in_cm3 = np.pi*(-np.power(dr*irad,2)+np.power(dr*(irad+1),2))*kpc_in_cm*kpc_in_cm*kpc_in_cm*2.0*maxlength
                                Grho_in_g_cm_3 = np.sum(Gm_in_g)/shellvol_in_cm3
                                Gnism_in_cm_3p = (0.78+0.22*Nebave*0.76)/proton_mass_in_g*Grho_in_g_cm_3
				tpi_in_sp = pidecay_fac/Gnism_in_cm_3p #pi decay time in s
				cregycut = np.sum(cregy_in_erg[cutr])
				Lgammap += cregycut/tpi_in_sp*betapi/nopi_per_gamma #in erg/s
				#print 'Gnism_in_cm_3p', Gnism_in_cm_3p
			Lgammapl = np.append(Lgammapl,Lgammap)
			Lgammal=np.append(Lgammal,np.sum(Lgammacut))
			enllist=np.append(enllist,np.sum(cregyl[cut]))
			Gmcutwl=np.append(Gmcutwl, Gmcutw)
			cregywl=np.append(cregywl,cregyw)
			Nebcutal=np.append(Nebcutal,Nebcuta)
			timel=np.append(timel,float(i)*0.98*1e6)
		Lgamma_l = (enllist[1:]-enllist[:-1])/((timel[1:]-timel[:-1])*sec_in_yr)/7.51e-16/pidecay_fac*betapi/nopi_per_gamma	
		#Consider CR density ~ 1eV/cm^3 and nsim ~ 1cm^-3 
		tpi_in_s_ideal=pidecay_fac
		vol_in_cm3 = np.pi*np.power(withinr,2)*kpc_in_cm*kpc_in_cm*kpc_in_cm*2.0*maxlength
		cregy_in_erg_ideal = 1.0*vol_in_cm3/erg_in_eV
		Lgamma_ideal = cregy_in_erg_ideal/tpi_in_s_ideal*betapi/nopi_per_gamma
		cregyaden_in_eV_cm_3 = cregywl*1e10*solar_mass_in_g*km_in_cm*km_in_cm*erg_in_eV/vol_in_cm3
		nism_in_cm_3 = Gmcutwl*1e10*2e33/vol_in_cm3*(0.78+0.22*Nebcutal*0.76)/proton_mass_in_g
		tpi_in_s_ave = pidecay_fac/nism_in_cm_3
		Lgammaave_nism_cre = cregywl*1e10*solar_mass_in_g*km_in_cm*km_in_cm/tpi_in_s_ave*betapi/nopi_per_gamma
                Lsfr = Kroupa_Lsf*1.0*solar_mass_in_g/sec_in_yr*cspeed_in_cm_s*cspeed_in_cm_s
		print 'Lgammal', Lgammal
		print 'Lgamma_l', Lgamma_l
		print 'Lgammapl', Lgammapl
		print 'Lgamma_ideal', Lgamma_ideal
		print 'nism_in_cm_3', nism_in_cm_3
		print 'cregyaden_in_eV_cm_3', cregyaden_in_eV_cm_3
		print 'Lgammaave_nism_cre', Lgammaave_nism_cre
		print 'Lsfr*2e-4', Lsfr*2e-4
		plt.plot(timel[1:]/1e6, Lgammal[1:], label='from particles')
		plt.plot(timel[1:]/1e6, Lgammapl[1:], label='ave cylinders')
		plt.plot(timel[1:]/1e6, Lgammaave_nism_cre[1:], label='ave whole')
		plt.plot(timel[1:]/1e6, np.absolute(Lgamma_l), label='from code')
		plt.axhline(y=Lgamma_ideal, label='ideal',ls='dashed')	
		plt.title(runtodo)
                plt.legend(loc='best')
                plt.ylabel(r'$L_{\gamma} {\rm (erg/s)}$')
                plt.xlabel('t (Myr)')
		plt.savefig('CRplot/Lgamma_ave_'+runtodo+'.pdf')
		plt.clf()
