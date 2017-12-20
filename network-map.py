###
#Script for NACOM network map
#Resources required: Shape files in /cb_2013_us_state_500k/, and /2012USoil.csv
#Author: Jimi Oke
#Created: Mar 11 2015
#Last mod: Jul 21 2016

import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.ion()
import matplotlib.font_manager as fm

#prop = fm.findfont('helvetica')

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

from operator import itemgetter, attrgetter

import brewer2mpl as b2m # Brewer colors; view here: http://bl.ocks.org/mbostock/5577023
set1H = b2m.get_map('Set1','Qualitative', 6).hex_colors # HEX values of Brewer Set1 colors
set1 = b2m.get_map('Blues','Sequential', 5).mpl_colors
set11 = b2m.get_map('Set1','Qualitative', 6).mpl_colors # HEX values of Brewer Set1 colors


# https://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi/23021198#23021198
def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length


# This generates Figure 4 in the paper. Run "usmap('rail')" or "usmap('pipe')" to show solely either of those modes
def usmap(mode):
 	#m = Basemap(projection='kav7',lon_0=0,resolution='c')
	m = Basemap(width=12000000,height=7000000,
		rsphere=(6378137.00,6356752.3142),\
		resolution='l',area_thresh=10000.,projection='lcc',\
		lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)

	# m = Basemap(projection='nsper',lon_0=-105,lat_0=40,
	#         satellite_height=3000*1000.,resolution='l')
	# m = Basemap(llcrnrlon=-170,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=71,
 #        projection='lcc',lat_1=33,lat_2=71,lon_0=-120)

	# # Cylindrical projection; makes ALASKA too large
	# m = Basemap(projection = 'cyl', llcrnrlat=22,urcrnrlat=71,\
	# 	llcrnrlon=-170,urcrnrlon=-64,resolution='c')
	#m.drawcoastlines(color='0.8',linewidth=0.5)
	#m.drawmapboundary(color='gray',linewidth=1)

	shp_info = m.readshapefile('./cb_2013_us_state_500k/cb_2013_us_state_500k','states',drawbounds=False)
	statenames=[]

	for shapedict in m.states_info:
		statename = shapedict['STUSPS']
		#colors[countryname] = initialu.index[initialu.ISO==countryname][0]
		statenames.append(statename)
	#print statenames
	#print m.states_info

	#plt.clf()
	# State centroids; source; http://dev.maxmind.com/geoip/legacy/codes/state_latlon/
	latlon = pd.read_csv('state_latlon.csv') # read in state centroids (latitutdes, longitudes)
	latlon = latlon.set_index('state')

	#Arcs2 does not include scenario pipelines (MI-NJ, MT-WA)
	network = pd.read_csv('./Primary-Data/Input-Parameters/Arcs2.csv')
	if mode=='pipe':
		network = network[network.type=='Pipeline']
	elif mode=='rail':
		network = network[network.type=='Rail']
		network.node_out = network.node_out.str.split('_').str.get(0)
		network.node_in = network.node_in.str.split('_').str.get(0)
	elif mode=='piperail': #plot both pipe and rail networks
		network = network[network.type.isin(['Pipeline','Rail'])]
		# print network
		for i in network.index:
			if network.ix[i,'type']=='Rail':
				nOut = str(network.ix[i,'node_out'])
				nIn = str(network.ix[i,'node_in'])
				network.ix[i,'node_out'] = nOut.split('_')[0]
				network.ix[i,'node_in']  = nIn.split('_')[0]				
	elif mode=='ship':
		network = network[network.type in ['Ship','BargeR','BargeS']]

	# railNet = pd.read_csv('rail.csv',header=None) # read in rail incidences
	# railNet.columns = ['lat','lon']


	# pipeNet = pd.read_csv('pipelines.csv',header=None) # read in pipeline incidences
   	oildata = pd.read_csv('./Primary-Data/2012USoil.csv')
	prod = oildata[[0,1]][0:9]
	refcon = oildata[[2,3]][0:18]
	petcon = oildata[[3,4]]
	imports = oildata[[5,6]]
	pmax = prod.Production.max(); pmin = prod.Production.min()
	rmax = refcon.RefConsumption.max(); rmin = refcon.RefConsumption.min()

	normProd = mpl.colors.Normalize(vmin=pmin, vmax=pmax+0.0025*(pmax-pmin))
	normRef = mpl.colors.Normalize(vmin=rmin, vmax=rmax)#+0.0625+(rmax-rmin))

	colors={}

	refColor = plt.cm.Oranges
	
	stateCentroids = dict()
	stateColor = dict()
	stateBox = dict()
	boxfc = dict()
	plt.axes()
	ax = plt.gca()
	fig = plt.gcf()
	stateColor['EC'] = '0.2'; stateColor['WC'] = '0.2'; stateColor['MX'] = '0.2'
	stateBox['EC'] = '0.5'; stateBox['WC'] = '0.5'; stateBox['MX'] = '0.5'
	boxfc['EC'] = 'w'; boxfc['WC'] = 'w'; boxfc['MX'] = 'w'

	#fig.set_size_inches(18,10)
	# cycle through country names, color each one.
	stateLabelColor = '0.2'
	stateLabelAlpha = '0.5'
	for nshape,seg in enumerate(m.states):
		name = statenames[nshape]
		#xx,yy = zip
		stateColor[name] = '0.2'
		stateBox[name] = '0.5'
		if name in prod.State.tolist():
			pval = float(prod.Production[prod.State == name]) # get production value
			cmap = plt.cm.Blues # color map
			colors[name] = cmap(np.sqrt((pval - pmin)/(pmax-pmin)) +.05 )[:3]
			color = rgb2hex(colors[name]) 
			ecolors={}
			boxfc[name] = "w"
			#mappableProd = plt.cm.ScalarMappable(cmap=cmap)
			#mappableProd.set_array([])
			#mappableProd.set_clim(pmin+.05, pmax)
			
			if name in refcon.State1.tolist():
				rval = float(refcon.RefConsumption[refcon.State1 == name]) # get production value
				cmap = refColor # color map
				ecolors[name] = cmap(np.sqrt( (rval - rmin)/(rmax-rmin)) +.25 )[:3]
				ecolor = rgb2hex(ecolors[name])
				pattern = '///'
				lab = ''#US node'
			else:
				ecolor = '0.8'
				pattern = None
				lab = ''#US node' #'Node: Producer'

			poly = Polygon(seg,closed=True,facecolor=set1[1],edgecolor=set1[4],label=lab,alpha=.8,joinstyle='round',
				zorder=3)
			if name == 'AK' and mode != 'ship':
				pass
			else:
				ax.add_patch(poly)
			# vertices = poly.get_xy()
			# centroid = centroidnp(vertices)
			# stateCentroids[name] = centroid

		elif name in refcon.State1.tolist():
			rval = float(refcon.RefConsumption[refcon.State1 == name]) # get production value
			cmap = refColor # color map
			colors[name] = cmap(np.sqrt( (rval - rmin)/(rmax-rmin))+.25 )[:3]
			color = rgb2hex(colors[name]) 
			ecolor = color
			pattern='///'
			lab = ''#US node' #'Node: Refiner'
			poly1 = Polygon(seg,closed=True,facecolor=set1[1],edgecolor=set1[4],label=lab,alpha=.8,joinstyle='round',
				zorder=3)
			if name == 'AK' and mode != 'ship':
				pass
			else:
				ax.add_patch(poly1)
			boxfc[name] = "w"
			# vertices = poly.get_xy()
			# centroid = centroidnp(vertices)
			# stateCentroids[name] = centroid
			#poly1.set_color(color)
		else:
			color = 'none'
			ecolor = '0.6'
			pattern= 'xxx'
			lab = ''
			poly = Polygon(seg,closed=True,facecolor=color,edgecolor=ecolor,hatch=pattern,label=lab,alpha=0.2,joinstyle='round',zorder=2)
			ax.add_patch(poly)
			stateColor[name] = 'none'
			stateBox[name] = 'none'
			boxfc[name] = 'none'
			# vertices = poly.get_xy()
			# centroid = centroidnp(vertices)
			# stateCentroids[name] = centroid

	#statePolygons = ax.get_patches()
	for centroid in latlon.index.tolist():
		if centroid == 'RW':
			pass
		else:
			xy = latlon.ix[centroid]
			x, y = m(xy[1], xy[0])
			m.scatter(x, y, color=stateColor[centroid],zorder=5000)
			ax.text(x-150000, y-50000, centroid, bbox = dict(boxstyle="round", fc=boxfc[centroid], ec=stateBox[centroid], alpha=0.5),
			 fontsize=10,color=stateColor[centroid]) #,zorder=300)

	if mode=='pipe':			
		for line in network.index:
			nodeOut = network.ix[line]['node_out']
			nodeIn = network.ix[line]['node_in']
			if nodeOut != nodeIn:
				coordsOut =  m(latlon.ix[nodeOut][1], latlon.ix[nodeOut][0])
				coordsIn =  m(latlon.ix[nodeIn][1], latlon.ix[nodeIn][0])
				x = (coordsOut[0], coordsIn[0])
				y = (coordsOut[1], coordsIn[1])
				m.plot(x, y, color = '0.6', lw = 4.5,zorder = 1000+int(line), alpha=0.6, label='Pipeline')
	elif mode=='rail':
		for line in network.index:
			nodeOut = network.ix[line]['node_out']
			nodeIn = network.ix[line]['node_in']
			if nodeOut != nodeIn:
				coordsOut =  m(latlon.ix[nodeOut][1], latlon.ix[nodeOut][0])
				coordsIn =  m(latlon.ix[nodeIn][1], latlon.ix[nodeIn][0])
				x = (coordsOut[0], coordsIn[0])
				y = (coordsOut[1], coordsIn[1])
				m.plot(x, y, '--', color = set11[4], lw = 1.4,zorder = 2000+int(line), alpha=0.9,label='Rail')
	elif mode=='piperail':
		# print network
		for line in network.index:
			nodeOut = network.ix[line]['node_out']
			nodeIn = network.ix[line]['node_in']
			if nodeOut != nodeIn:
				coordsOut =  m(latlon.ix[nodeOut][1], latlon.ix[nodeOut][0])
				coordsIn =  m(latlon.ix[nodeIn][1], latlon.ix[nodeIn][0])
				x = (coordsOut[0], coordsIn[0])
				y = (coordsOut[1], coordsIn[1])
				if network.ix[line]['type'] == 'Pipeline':
					m.plot(x, y, color = '0.6', lw = 4.7,zorder = 1000+int(line), alpha=0.5, label='Pipeline')
				elif network.ix[line]['type'] == 'Rail':
					m.plot(x, y, '--', color = set11[4], lw = 1.4,zorder = 2000+int(line), alpha=0.9,label='Rail')
	elif mode=='water':
		for line in network.index:
			nodeOut = network.ix[line]['node_out']
			nodeIn = network.ix[line]['node_in']
			if nodeOut == 'RW' and nodeIn in ['AK','CA','WA']:
				latlon.ix['RW'][0] = 37.930029 
				latlon.ix['RW'][1] = -144.180625	
			elif nodeOut == 'RW' and nodeIn in ['AL','MS','TN','LA','TX','MX']:
				latlon.ix['RW'][0] = 18.274893
				latlon.ix['RW'][1] = -75.054649
			elif nodeOut == 'MX' and nodeIn in 'RW':
				latlon.ix['RW'][0] = 18.274893
				latlon.ix['RW'][1] = -75.054649
			else:
				latlon.ix['RW'][0] = 35.248893 
				latlon.ix['RW'][1] = -55.411094
			coordsOut =  m(latlon.ix[nodeOut][1], latlon.ix[nodeOut][0])
			coordsIn =  m(latlon.ix[nodeIn][1], latlon.ix[nodeIn][0])
			x = (coordsOut[0], coordsIn[0])
			y = (coordsOut[1], coordsIn[1])
			m.plot(x, y, color = set11[0], lw = 1.8,zorder = 3000+int(line), alpha=0.6,label='Ship')
		#m.drawgreatcircle(x[0], y[0], x[1], y[1],lw=2,c=set11[1])
	#pipeLabel = plt.plot([], [], c = 'gray', lw = 1.5)
	#railLabel = plt.plot([], [], '--', c = 'r', lw = 1.5)
	#plt.legend([pipeLabel,railLabel],labels=['Pipeline','Rail'],loc='lower left',fontsize=12)

	handles,labels = ax.get_legend_handles_labels()
	lis = []
	seen =set()
	for i in np.arange(len(handles)):
		value = (handles[i],labels[i])
		lis.append(value)

	lis1 = [item for item in lis if item[1] not in seen and not seen.add(item[1])]
	lis1 = sorted(lis1,key=itemgetter(1))
	handles1 = [i[0] for i in lis1]
	labels1 = [i[1] for i in lis1]
	ax.legend(handles1,labels1, bbox_to_anchor=(0.4, 0.22),handlelength=4, framealpha=0.5,frameon=False,fontsize=14) #, ncol = len(labels1))

	plt.show()


if __name__ == '__main__':
	usmap('piperail')