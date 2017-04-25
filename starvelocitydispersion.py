from readsnap_samson import *
from Sasha_functions import *
import sys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from gadget_lib.cosmo import *
from samson_functions import *
#dirneed=['383','g383_nag','g0320_383','g0301_383']
withinkpc=20.0
#reff=3.0
#reff=1.98
#r12 = reff*4.0/3.0
dirneed=['f573','f553','f476','fm11','f383','f61']
#tlist=[[13.4,8.3,13.5],[10.4,5.8,13.5],[10.7,2.3,13.5],[2.7,2.0,6.0],[2.0,2.0,6.5],[2.0,2.0,2.4]]
tlist= [[13.4,8.2,13.5],[10.4,5.7,13.5],[10.7,2.6,13.5],[2.7,2.3,6.0],[2.0,2.0,6.5],[2.2,2.2,2.4]]
#Nsnap=234
#Nsnap=157
#dirneed=['553','476']
def stellarveldis(runtodo,time):
	rlist=np.linspace(0.1,20,num=200)
	mlist=[]
        snaplist, timelist = readtime()
        Nsnap = int(np.interp(time, timelist, snaplist))
	haloinfo=cosmichalo(runtodo)
	rundir=haloinfo['rundir']
	subdir=haloinfo['subdir']
	maindir=haloinfo['maindir']
	multifile=haloinfo['multifile']
	halocolor=haloinfo['halocolor']
	halostr=haloinfo['halostr']
        halosA = read_halo_history(rundir, halonostr=halostr,maindir=maindir)
        redlist = halosA['redshift']
        xlist = halosA['x']
        ylist = halosA['y']
        zlist = halosA['z']
        halolist =halosA['ID']
        mvirlist = halosA['M']
        alist = np.array(1./(1.+np.array(redlist)))
        xlist = np.array(xlist)*np.array(alist)/0.702
        ylist = np.array(ylist)*np.array(alist)/0.702
        zlist = np.array(zlist)*np.array(alist)/0.702
	print 'halolist', halolist


	if (int(Nsnap) < 10):
		Nsnapstring = '00'+str(Nsnap)
	elif (int(Nsnap) < 100):
		Nsnapstring = '0'+str(Nsnap)
	else:
		Nsnapstring = str(Nsnap)

	the_snapdir = '/home/tkc004/'+maindir+'/'+rundir+'/'+subdir
	the_prefix ='snapshot'
	the_suffix = '.hdf5'
	header = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, header_only=1, h0=1,cosmological=1)
	S = readsnap(the_snapdir, Nsnapstring, 4, snapshot_name=the_prefix, extension=the_suffix, h0=1,cosmological=1)
	ascale = header['time']
	thisz = 1./ascale-1.
	print 'thisz', thisz
	hubble = header['hubble']
	print 'hubble', hubble
	xcen=np.interp(np.log(ascale),np.log(alist),xlist) #physical distance (kpc)
	ycen=np.interp(np.log(ascale),np.log(alist),ylist)
	zcen=np.interp(np.log(ascale),np.log(alist),zlist)
	halono=np.interp(np.log(ascale),np.log(alist),halolist)
	mvir = np.interp(np.log(ascale),np.log(alist),mvirlist)
	print 'cen', xcen, ycen, zcen

	Spos = S['p'][:,:]
	Smass = S['m'][:]
	Svel = S['v'][:,:]
	Sx = Spos[:,0]
	Sy = Spos[:,1]
	Sz = Spos[:,2]
	Svx=Svel[:,0]
	Svy=Svel[:,1]
	Svz=Svel[:,2]
	Sr = np.sqrt(np.square(Sx-xcen)+np.square(Sy-ycen)+np.square(Sz-zcen))
	Srx = np.sqrt(np.square(Sy-ycen)+np.square(Sz-zcen))
        Sry = np.sqrt(np.square(Sx-xcen)+np.square(Sz-zcen))
        Srz = np.sqrt(np.square(Sx-xcen)+np.square(Sy-ycen))
	for i in range(len(rlist)):
		withinrlist = Sr<rlist[i]
		withinrm = np.sum(Smass[withinrlist]*1e10)
		mlist = np.append(mlist,withinrm)
	r12 = np.interp(mlist[-1]*0.5,mlist,rlist)
	reff = r12*3.0/4.0
	withinr = Sr<r12
	withinx = Srx<withinkpc
        withiny = Sry<withinkpc
        withinz = Srz<withinkpc
	Swithinm = np.sum(Smass[withinr]*1e10)
	print 'rlist', rlist[::10]
	print 'mlist', mlist[::10]
	print 'reff', reff
	xdis=np.std(Svx[withinx])
	ydis=np.std(Svy[withiny])
	zdis=np.std(Svz[withinz])
	print 'x dispersion', xdis
        print 'y dispersion', ydis
        print 'z dispersion', zdis
	print 'average dispersion', (np.std(Svx[withinx])+ np.std(Svy[withinx])+ np.std(Svz[withinx]))/3.0
	print 'm1/2 from sigmax', np.power(np.std(Svx[withinx]),2)*9.3e5*reff
        print 'm1/2 from sigmay', np.power(np.std(Svy[withiny]),2)*9.3e5*reff
        print 'm1/2 from sigmaz', np.power(np.std(Svz[withinz]),2)*9.3e5*reff
	print 'average m1/2', np.power((np.std(Svx[withinx])+ np.std(Svy[withinx])+ np.std(Svz[withinx]))/3.0,2)*9.3e5*reff
	del S, Spos, Smass, Sx, Sy, Sz, Svx, Svy, Svz, Sr, Srx, Sry, Srz
	G = readsnap(the_snapdir, Nsnapstring, 0, snapshot_name=the_prefix, extension=the_suffix, cosmological=1, h0=1)
	Gpos = G['p'][:,:]
	Gmass = G['m'][:]
	Gx = Gpos[:,0]
	Gy = Gpos[:,1]
	Gz = Gpos[:,2]
	Gr = np.sqrt(np.square(Gx-xcen)+np.square(Gy-ycen)+np.square(Gz-zcen))
	withinr = Gr<r12
	Gwithinm = np.sum(Gmass[withinr]*1e10)
	del G, Gpos, Gmass, Gx, Gy, Gz, Gr
        DM = readsnap(the_snapdir, Nsnapstring, 1, snapshot_name=the_prefix, extension=the_suffix, cosmological=1, h0=1)
	DMpos = DM['p'][:,:]
	DMmass = DM['m'][:]
	DMx = DMpos[:,0]
	DMy = DMpos[:,1]
	DMz = DMpos[:,2]
	DMr = np.sqrt(np.square(DMx-xcen)+np.square(DMy-ycen)+np.square(DMz-zcen))
	withinr = DMr<r12
	DMwithinm = np.sum(DMmass[withinr]*1e10) #in solar mass
	masswithin = Gwithinm+Swithinm+DMwithinm
	print 'masswithin', masswithin
	print 'average density', masswithin/reff/reff/reff
	print 'log(masswithin)', np.log10(masswithin)
	print 'expected dispersion', np.sqrt(masswithin/9.3e5/reff)
	return reff, xdis, ydis, zdis, masswithin

adstr=' '
Mexpstr=' '
M12str=' '
for ncount in range(len(dirneed)):
        runtodo = dirneed[ncount]
        refflist=[]
        xdislist=[]
        ydislist=[]
        zdislist=[]
	minlist=[]
	time=tlist[ncount][2]
	reff, xdis, ydis, zdis, masswithin=stellarveldis(runtodo,time)
	avedis=(xdis+ydis+zdis)/3.0
        adround = np.around(avedis, decimals=1)
	dmin = np.around(np.amin([xdis,ydis,zdis]), decimals=1)
        dmax = np.around(np.amax([xdis,ydis,zdis]), decimals=1)
        adstr = adstr+'&      $'+str(adround)+'$& $\,_{'+str(dmin)+'}^{'+str(dmax)+'} $'
	Mexp=np.power(avedis,2)*9.3e5*reff
	Mexpround = np.around(Mexp/1e9, decimals=2)
	Mexpstr = Mexpstr+'&\multicolumn{2}{l}{'+str(Mexpround)+'}'
        M12round = np.around(masswithin/1e9, decimals=2)
        M12str = M12str+'&\multicolumn{2}{l}{'+str(M12round)+'}'
print 'ave dis', adstr+'\\\\'
print 'Mexp', Mexpstr+'\\\\'
print 'M12', M12str+'\\\\'
	
