'''
Methods for generating charts/maps for NACOM 
(as shown in "Multimodal flows in energy networks..." paper)
Author: Jimi Oke
Created: Mar 4 2016
Last Mod: Jul 21 2016
Note: may need to install basemap package to run
'''
import pandas as pd
import numpy as np
import math
import os
import matplotlib.transforms as transforms
from matplotlib.patches import ConnectionPatch

# import xlrd
import matplotlib.pyplot as plt
#plt.ion()
import itertools
import matplotlib as mpl
import matplotlib.font_manager as fm

from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D 

mpl.rc('text',usetex=True)

from operator import itemgetter, attrgetter

import brewer2mpl as b2m # Brewer colors; view here: http://bl.ocks.org/mbostock/5577023

import sys
sys.path.insert(0,'./Primary-Data/')
import dataMethods
nicered = "#E6072A"

set1H = b2m.get_map('Set1','Qualitative', 6).hex_colors # HEX values of Brewer Set1 colors
blues = b2m.get_map('Blues','Sequential', 5).mpl_colors
set11 = b2m.get_map('Set1','Qualitative', 6).mpl_colors # HEX values of Brewer Set1 colors
set2 = b2m.get_map('Set2','Qualitative', 3).mpl_colors # HEX values of Brewer Set1 colors


#******************************************************************************************
#********************************   F i g u r e   2  *************************************
#******************************************************************************************
def prodChart():
	p = dataMethods.production()
	p = p[p.supplier!='RW']
	A = map(int, p.ref[p.fuel=='HE_SO_CRUDE'].tolist() )
	B = map(int, p.ref[p.fuel=='LI_SW_CRUDE'].tolist() )
	#X = map(str, p.node.unique().tolist() )
	nodeNames = [dataMethods.names.node_name[dataMethods.names.node_abb==x].tolist() for x in p.node.unique()]
	nodeNames = list(itertools.chain.from_iterable(nodeNames))	

	lightGreen = '#a1d99b'
	heavyGreen = '#31a354'
	fig = plt.figure(figsize=(15,10))
	ax = fig.add_subplot(111)
	barw = .8
	ypos = np.flipud(np.arange(len(A)))
	ax.barh(ypos, A, barw, color=heavyGreen, edgecolor ='none')
	ax.barh(ypos, B, barw, color=lightGreen, edgecolor ='none', left=A)

	ax.set_yticks(ypos + barw/2.)
	ax.set_yticklabels(nodeNames, fontsize=22)#,rotation=45)
	ax.set_xlabel('Quantity (mpbd)',ha='left',fontsize=20)
	ax.xaxis.set_label_coords(1.05,-0.035)
	ax.set_xticklabels([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5])
	#ax.tick_params(which='both',bottom='off',top='off' width=2, colors='r')
	ax.tick_params(direction='out', pad=3,labelsize=22)

	#ax.xaxis.tick_top()
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_frame_on(False)

	# ax.set_title('Crude oil production', fontsize=24,ha='left')
	# plt.subplots_adjust(top=0.9)
	# ax.title.set_position((0,1.08))
	#plt.axis('off')

	#LEGEND
	l1 = Line2D([], [], linewidth=6, color=lightGreen) 
	l2 = Line2D([], [], linewidth=6, color=heavyGreen) 	

	labels = ['Light-sweet crude', 'Heavy-sour crude']
	leg = ax.legend([l1,l2], labels, ncol=1, frameon=False, fontsize=22,
		           bbox_to_anchor=[1.3, 0.95], handlelength=2, 		#0.11
			        handletextpad=1, columnspacing=2, title='Fuel type')

	plt.setp(leg.get_title(), fontsize=22)
	leg.get_title().set_position((0, 10))

	plt.savefig('production.pdf',bbox_inches='tight',dpi=900, transparent=True)
	plt.show()



#******************************************************************************************
#********************************   F i g u r e   3  *************************************
#******************************************************************************************
def conChart():
	c = dataMethods.consumption()
	c = c[c.node!='RW']

	A = map(int, c.value[c.fuel=='HE_SO_CRUDE'].tolist() )
	B = map(int, c.value[c.fuel=='LI_SW_CRUDE'].tolist() )
	#X = map(str, p.node.unique().tolist() )
	nodeNames = [dataMethods.names.node_name[dataMethods.names.node_abb==x].tolist() for x in c.node.unique()]
	nodeNames = list(itertools.chain.from_iterable(nodeNames))	

	fig = plt.figure(figsize=(15,13))
	ax = fig.add_subplot(111)
	barw = .8
	ypos = np.flipud(np.arange(len(A)))
	ax.barh(ypos, A, barw, color='#3182bd',edgecolor ='none')
	ax.barh(ypos, B, barw, color='#9ecae1',edgecolor ='none',left=A)

	ax.set_yticks(ypos + barw/2.)
	ax.set_yticklabels(nodeNames,fontsize=22)#,rotation=45)
	ax.set_xlabel('Quantity (mbpd)',ha='left',fontsize=20)
	ax.xaxis.set_label_coords(1.05,-0.035)
	ax.set_xticklabels([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0])
	#ax.tick_params(which='both',bottom='off',top='off' width=2, colors='r')
	ax.tick_params(direction='out', pad=3,labelsize=22)

	#ax.xaxis.tick_top()
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_frame_on(False)

	# ax.set_title('Crude oil demand', fontsize=24,ha='left')
	# plt.subplots_adjust(top=0.9)
	# ax.title.set_position((0,1.08))
	#plt.axis('off')

	#LEGEND
	l1 = Line2D([], [], linewidth=6, color='#9ecae1') 
	l2 = Line2D([], [], linewidth=6, color='#3182bd') 	

	labels = ['Light-sweet crude', 'Heavy-sour crude']
	leg = ax.legend([l1,l2], labels, ncol=1, frameon=False, fontsize=22,
		           bbox_to_anchor=[1.3, 0.95], handlelength=2, 		#0.11
			        handletextpad=1, columnspacing=2, title='Fuel type')

	plt.setp(leg.get_title(), fontsize=22)
	leg.get_title().set_position((0, 10))

	plt.savefig('consumption.pdf',bbox_inches='tight',dpi=900, transparent=True)
	plt.show()


def getFlows(scenario):
	flow = pd.read_excel('./ProdFlows_'+scenario+'.xlsx','Flows') # read flows from ProdFlows (linked to mfm_report)
	flow = flow.dropna() # drop zero values
	flow = flow[flow.disaggregation!='region'] # 10/29/15 - new format
	flow = flow[flow.values!=-1] # remove non-flow (likely redundant)
	flow = flow[flow.type=='flow'] # consider only flows not caps
	flow = flow[flow.season=='Y'] # restrict to one season to avoid dupes

	# Auxiliary rail nodes are denoted "*_R". Removes this suffix for ease of plotting
	flow.node_out = flow.node_out.str.split('_').str.get(0)
	flow.node_in = flow.node_in.str.split('_').str.get(0)

	flow.drop_duplicates(inplace=True)  # drop duplicate values
	return flow


nodes = pd.read_excel('./Primary-Data/SearchList_NodeRegionNames.xlsx','Nodes') # read node names
nodes = nodes[[0,1,2,6]].dropna() # ick up 
nodes = nodes[nodes.Consumer==1]

# helper funcs for label rotations
# http://stackoverflow.com/a/18800233/3023033
rotated_labels = []
def text_slope_match_line(text, x, y, line):
	global rotated_labels

	# find the slope
	xdata, ydata = line.get_data()

	x1 = xdata[0]
	x2 = xdata[-1]
	y1 = ydata[0]
	y2 = ydata[-1]

	rotated_labels.append({"text":text, "line":line, "p1":np.array((x1, y1)), "p2":np.array((x2, y2))})

def update_text_slopes():
	global rotated_labels

	for label in rotated_labels:
		# slope_degrees is in data coordinates, the text() and annotate() functions need it in screen coordinates
		text, line = label["text"], label["line"]
		p1, p2 = label["p1"], label["p2"]

		# get the line's data transform
		ax = line.get_axes()

		sp1 = ax.transData.transform_point(p1)
		sp2 = ax.transData.transform_point(p2)

		rise = (sp2[1] - sp1[1])
		run = (sp2[0] - sp1[0])

		slope_degrees = math.degrees(math.atan(rise/run))

		text.set_rotation(slope_degrees)


def outflows(scenario, country):
	flow = getFlows(scenario)
	df = flow[flow.scenario == scenario]
	if country=='can':
		df = df[df.node_out.isin(['WC','EC'])]
	elif country=='usa':
		df = df[~df.node_out.isin(['WC','EC','MX'])]
		df = df[df.node_in.isin(['WC','EC','RW','MX'])]
	elif country=='mex':
		df = df[df.node_out=='MX']
	#df = df.set_index(['year','arctype'])
	df[df.node_out != 'EC']
	df[df.node_out != 'MX'] 
	df[df.node_out != 'RW'] 
	df[df.node_out != 'WC']
	df[df.node_in != 'RW'] 
	df[df.node_in != 'EC']
	df[df.node_in != 'MX']
	df[df.node_in != 'WC']

	grouped = df.groupby(['year','arctype'])['value'].agg({'value' : np.sum})
	grouped = grouped.reset_index()
	grouped = grouped.pivot(index='year',columns='arctype',values='value')
	grouped = grouped.fillna(0)
	for a in ['BargeR','BargeS','Ship']:
		if a not in grouped.columns:
			grouped[a] = 0
	grouped['Ship/Barge'] = grouped['Ship'] + grouped['BargeS'] + grouped['BargeR']
	try:
		grouped = grouped[['Rail','Pipeline','Ship/Barge']]
	except KeyError:
		try:
			grouped = grouped[['Pipeline','Ship/Barge']]
		except KeyError:
			grouped = grouped[['Ship/Barge']]
	if scenario != 'base':
		grouped.ix[2012] = 0
	return df, grouped


def scenarioflows(scenario):
	flow = getFlows(scenario)
	df = flow[flow.scenario == scenario]
	#df = df.set_index(['year','arctype'])
	df[df.node_out != 'EC']
	df[df.node_out != 'MX'] 
	df[df.node_out != 'RW'] 
	df[df.node_out != 'WC']
	df[df.node_in != 'RW'] 
	df[df.node_in != 'EC']
	df[df.node_in != 'MX']
	df[df.node_in != 'WC']

	grouped = df.groupby(['year','arctype'])['value'].agg({'value' : np.sum})
	grouped = grouped.reset_index()
	grouped = grouped.pivot(index='year',columns='arctype',values='value')
	grouped = grouped.fillna(0)
	for a in ['BargeR','BargeS','Ship']:
		if a not in grouped.columns:
			grouped[a] = 0
	grouped['Ship/Barge'] = grouped['Ship'] + grouped['BargeS'] + grouped['BargeR']
	grouped = grouped[['Rail','Pipeline','Ship/Barge']]
	if scenario != 'base':
		grouped.ix[2012] = 0
	return grouped



## Stacked scenario bars
## Function obtained from
#http://stackoverflow.com/a/22845857/3023033
def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
	"""Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
	labels is a list of the names of the dataframe, used for the legend
	title is a string for the title of the plot
	H is the hatch used for identification of the different dataframe"""

	n_df = len(dfall)
	n_col = len(dfall[0].columns) 
	n_ind = len(dfall[0].index)
	axe = plt.subplot(111)

	for df in dfall : # for each data frame
		df = df/1000.
		axe = df.plot(kind="bar",
					  linewidth=0,
					  stacked=True,
					  ax=axe,
					  legend=False,
					  grid=False,
					  color=set11,
					  **kwargs)  # make bar plots
	patterns = ['None','None','None','/','/','/','\\','\\','\\','-','-','-','x','x','x']
	h,l = axe.get_legend_handles_labels() # get the handles we want to modify
	for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
		for j, pa in enumerate(h[i:i+n_col]):
			print j, pa
			for rect in pa.patches: # for each index
				print rect
				if i==0 and rect.get_x()==0.25:
					rect.set_x(rect.get_x() + 1 / float(n_df + 1) * (i+12) / float(n_col))
				else:
					rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
				print rect.get_x()
				if i == 0:
					pass
				else:
					rect.set_hatch(patterns[i])
				rect.set_width(1 / float(n_df + 1))
				rect.set_edgecolor("w")
		#patternInd += 1
	plt.rc('text', usetex=True)
	xtickPos = (np.arange(0, 2 * n_ind, 2) + 8/ float(n_df + 1) ) / 2.
	xtickPos[0] = 12/6./2.
	axe.set_xticks(xtickPos) #x tick positions
	axe.set_xticklabels(df.index, rotation = 0,fontsize=24)
	axe.set_xlabel("")
	axe.set_yticklabels(np.arange(0,25,5), fontsize=24)
	axe.set_ylabel('Quantity (mbpd)',fontsize=24)
	axe.set_title(title,fontsize=26)

	axe.spines['right'].set_visible(False)
	axe.spines['top'].set_visible(False)
	axe.spines['bottom'].set_visible(False)
	axe.spines['left'].set_visible(False)
	#axe.spines['bottom'].set_color('gray')

	axe.tick_params(axis='both', which='major', left='off', bottom='off',top='off',right='off')
	axe.yaxis.set_tick_params(pad=15)
	# Only show ticks on the left and bottom spines
	#axe.yaxis.set_ticks_position('left')
	#axe.xaxis.set_ticks_position('bottom')

	for yy in np.arange(0,25,5):
		opac = 0.4
		zOrd = 0
		axe.axhline(y=yy, linewidth=4, alpha = opac, color='gray', zorder=zOrd)

	# Add invisible data to add another legend
	n=[]  
	for i in range(n_df):
		#n.append(axe.bar(0, 0, color="gray", edgecolor="w",hatch=H * i))
		if i == 0:
			n.append(axe.bar(0, 0, color="gray", edgecolor="w"))
		else:
			n.append(axe.bar(0, 0, color="gray", edgecolor="w", hatch=patterns[i*n_col]))

	#legend for modes 
	l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.02, 0.7], handleheight=2, handlelength=4, fontsize=24,frameon=False)

	#legend for scenarios
	if labels is not None:
		l2 = plt.legend(n, labels, loc=[-.05, -0.25], handleheight=3, handlelength=4, fontsize=24,ncol= 3)#,mode='expand') 
		#l2 = plt.legend(n, labels, loc=[-.05, -0.15], handleheight=3, handlelength=4, fontsize=16,ncol= n_df)#,mode='expand') 
		#l2 = plt.legend(n, labels, loc=[1.01, 0.2], handleheight=2, handlelength=5, fontsize=18,ncol= 1)#,mode='') 
	axe.add_artist(l1)
	axe.set_ylim([0,25])
	xlimUpper = axe.get_xlim()[1]
	axe.set_xlim([0.75,xlimUpper])
	axe.lines[0].set_visible(False)
	return axe

dfBase = scenarioflows('base')
dfBase = dfBase.sort_index()
dfBan = scenarioflows('US_export_ban_lifted')
dfBan = dfBan.sort_index()
dfPipe = scenarioflows('US_midwest_pipelines')
dfPipe = dfPipe.sort_index()
dfCap = scenarioflows('bakken_rail_cap')
dfCap = dfCap.sort_index()
dfBanP = scenarioflows('US_ban_pipelines')
dfBanP = dfBanP.sort_index()



#******************************************************************************************
#********************************   F i g u r e   8  *************************************
#******************************************************************************************
# To achieve hatch line thickness as shown in the paper, you may need to open the matplotlib
# /backends/backend_pdf.py file, and in the writeHatches method modify the "output" attribute:
# self.output(0.1, Op.setlinewidth) --> self.output(2, Op.setlinewidth)
def barFlows():
	plot_clustered_stacked([dfBase, dfCap, dfPipe, dfBan, dfBanP],
		['Base Case','Capping Bakken Rail Flows', 'US Midwest Pipeline Investments', 'US Oil Export Ban Lifted', 'US Exports + Midwest Pipelines + Bakken Rail Caps'],
		title = "")
		#"Annual intra-U.S. multimodal crude oil flows by scenario")
	# plot_clustered_stacked([dfBase, dfPipe, dfCap],
	# 	['Base', 'Capping Flows From Bakken Region', 'US Midwest Pipeline Investments'],
	# 	title = "Annual intra-US multimodal crude oil flows by scenario")
	# 	#ecolor="none")
	plt.show()	



#******************************************************************************************
#********************************   F i g u r e s  5 - 7  ********************************* 
#******************************************************************************************
## Function generates flow maps
def usmap(scenario,mode,yyyy,save='n'):
	"""
	scenario: 'base', 'bakken_rail_cap', 'US_midwest_pipelines', 'US_export_ban_lifted'
	mode: 'Rail', 'Pipeline'
	years: 2012, 2015, 2018
	save: 'y', 'n'
	"""
	m = Basemap(width=12000000,height=7000000,
		rsphere=(6378137.00,6356752.3142),\
		resolution='l',area_thresh=1000.,projection='lcc',\
		lat_1=48.,lat_2=52,lat_0=50,lon_0=-107.)

	# m.drawcoastlines(color='gray',linewidth=0.5)
	m.drawmapboundary(color='w',linewidth=0,fill_color=None)

	shp_info = m.readshapefile('./cb_2013_us_state_500k/cb_2013_us_state_500k','states',drawbounds=False)
	statenames=[]

	for shapedict in m.states_info:
		statename = shapedict['STUSPS']
		statenames.append(statename)
	usanodes = pd.Series(statenames).unique()

	# State centroids; source; http://dev.maxmind.com/geoip/legacy/codes/state_latlon/
	latlon = pd.read_csv('latlon.csv') # read in state centroids (latitutdes, longitudes)
	latlon = latlon.set_index('state')
	#print latlon
	
	stateColor = dict()
	stateBox = dict()
	boxfc = dict()
	plt.axes()
	ax = plt.gca()
	fig = plt.gcf()
	stateColor['EC'] = '0.2'; stateColor['WC'] = '0.2'; stateColor['MX'] = '0.2'
	stateBox['EC'] = '0.5'; stateBox['WC'] = '0.5'; stateBox['MX'] = '0.5'
	boxfc['EC'] = 'w'; boxfc['WC'] = 'w'; boxfc['MX'] = 'w'

	# cycle through country names, color each one.
	stateLabelColor = '0.2'
	stateLabelAlpha = '0.5'
	for nshape,seg in enumerate(m.states):
		name = statenames[nshape]
		stateColor[name] = '0.2'
		stateBox[name] = '0.5'
		boxfc[name] = 'w'

		if name in nodes.node_abb.tolist():
			if name in ['AK','HI','PR','VI']:
				pass
			else: # Fill producing/consuming nodes in BLUE, 
				#cmap = plt.cm.Blues # color map
				ecolors={}
				pattern = None
				lab = 'US crude oil producer/refiner' #Node: Producer and/or Refiner
				poly = Polygon(seg,closed=True,facecolor=blues[1],edgecolor=blues[4],label=lab,alpha=.4,joinstyle='round',
					zorder=3)
				ax.add_patch(poly)
		else:
			if name in ['AK','HI','PR','VI']:
				pass
			else:
				color = 'w'
				ecolor = '0.7'
				pattern= 'xxx'
				lab = 'US state'
				poly = Polygon(seg,closed=True,facecolor=color,edgecolor=ecolor,hatch=pattern,label=lab,alpha=0.4,joinstyle='round',zorder=2)
				ax.add_patch(poly)
				stateColor[name] = 'none'
				stateBox[name] = 'none'
				boxfc[name] = 'none'


	# scenario and arc selection
	flow = getFlows(scenario)
	net = flow[flow.year==yyyy]
	net = net[net.scenario==scenario]
	net = net[net.arctype==mode]
	net = net.set_index(['arc','fuel'])
	print net
	#Initialize zorder param
	z = 1 
	nodeUsage = dict()
	nodeSize = dict()
	# shift the object over 2 points, and down 2 points
	dx, dy = 2/72., -3/72.
	adx, ady = 4/72., -8/72.  #arrow offsets
	offsetL = transforms.ScaledTranslation(dx, -dy, fig.dpi_scale_trans)
	offsetH = transforms.ScaledTranslation(-dx, dy, fig.dpi_scale_trans)
	aoffsetL = transforms.ScaledTranslation(adx, -ady, fig.dpi_scale_trans) 
	aoffsetH = transforms.ScaledTranslation(-adx, ady, fig.dpi_scale_trans)
	transformL = ax.transData + offsetL
	transformH = ax.transData + offsetH
	atransformL = ax.transData + aoffsetL
	atransformH = ax.transData + aoffsetH
	for node in latlon.index.unique():
		nodeUsage[node] = 0
		nodeSize[node] = 60 #40
	for arc in net.index.get_level_values(0).unique(): # for some reason indices duplicated
		#fprint arc
		try:
			nodeOut = net.ix[(arc,'HE_SO_CRUDE')].node_out
			nodeIn = net.ix[(arc,'HE_SO_CRUDE')].node_in
		except KeyError:
			nodeOut = net.ix[(arc,'LI_SW_CRUDE')].node_out
			nodeIn = net.ix[(arc,'LI_SW_CRUDE')].node_in
		try:	
			valH = net.ix[(arc,'HE_SO_CRUDE')].value
		except KeyError:
			valH = 0 
		try:
			valL = net.ix[(arc,'LI_SW_CRUDE')].value
		except KeyError:
			valL = 0
		annoval = int(valH) + int(valL)
		if valH == 0:
			arclabel = 'L:'+str(int(valL))
		elif valL == 0:
			arclabel = 'H:'+str(int(valL))		
		else:	
			arclabel = 'H:'+str(int(valH))+' L:'+str(int(valL))
		coordsOut =  m(latlon.ix[nodeOut][1], latlon.ix[nodeOut][0])
		coordsIn =  m(latlon.ix[nodeIn][1], latlon.ix[nodeIn][0])
		nodeUsage[nodeOut] = 1
		nodeUsage[nodeIn] = 1
		if nodeSize[nodeOut] < annoval:
			nodeSize[nodeOut] = annoval
		if nodeSize[nodeIn] < annoval:
			nodeSize[nodeIn] = annoval
		x = (coordsOut[0], coordsIn[0])
		y = (coordsOut[1], coordsIn[1])

		if x[1] >= x[0]:
			valcolor = 'g'
		else:
			valcolor = 'r'
		# Initialize coordinates for value labels
		xmid = (x[0] + x[1])/2
		ymid = (y[0] + y[1])/2
		thickness = .8*np.sqrt(annoval)
		thicknessL = .7*np.sqrt(valL)
		thicknessH = .7*np.sqrt(valH)
		## previously: lw = .01*annoval
		dx,dy = 0.3*(x[1]-x[0]), 0.3*(y[1]-y[0])
		if x[1] >= x[0]:
			arrShapeL, arrShapeH = 'right', 'left'
		else:
			arrShapeL, arrShapeH = 'left', 'right'
		if mode == 'Pipeline' or mode == 'Rail':
			lin,=m.plot(x,y,color = set2[1],   transform=transformL, lw = thicknessL, zorder = 4, alpha=.9 ) #, label='Light crude')
			lin,=m.plot(x,y, '--',color = set2[0],   transform=transformH, lw = thicknessH, zorder = 5, alpha=.9) #, label='Heavy crude' )
			if valL != 0: 
				plt.arrow(x[0],y[0],dx,dy,color = set2[1],   transform = atransformL,  zorder = 4, 
					alpha=.9,head_width=8000*thicknessL,head_length=8000*thicknessL,
					shape=arrShapeL, )
			if valH != 0:
				plt.arrow(x[0],y[0],dx,dy,color = set2[0],   transform = atransformH,  zorder = 5, 
					alpha=.9,head_width=8000*thicknessH,head_length=8000*thicknessH,
					shape=arrShapeH, )
		else:
			lin, = m.plot(x, y, color = set11[0],lw = 1, zorder = 100+z, alpha=0.7, label='Arc: %s' %mode)
		z = z + 1 
	for centroid in latlon.index.tolist():
		if centroid in ['HI','PR','VI']:
			pass
		else:
			if nodeUsage[centroid]==1:
				xy = latlon.ix[centroid]
				x, y = m(xy[1], xy[0])

				m.scatter(x, y, marker = 'o', color='w', s = 78*np.sqrt(nodeSize[centroid]),zorder=6,edgecolors='gray',alpha=.8)
				xoff = 100
				yoff = 100
				ax.annotate(centroid, (x,y),zorder=6,ha ='center',va='center',fontsize=16) # bbox = dict(boxstyle="round", fc=boxfc[centroid], ec=stateBox[centroid], alpha=0.5),
			else:
				pass

	xlimLU = ax.get_xlim()
	ylimLU = ax.get_ylim()
	ax.set_xlim([4500000,9100000])
	ax.set_ylim([250000,5000000])

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
	leg1=ax.legend(handles1,labels1,bbox_to_anchor=[0.43,.92],handlelength=3, framealpha=0.5,handletextpad=1, #0.225,.16 #orig .5
		frameon=False,fontsize=22) #, ncol = len(labels1))
	
	# Legend
	# Create fake labels for legend
	a = 0.7
	lwL1 = net[net.index.get_level_values('fuel')=='LI_SW_CRUDE'].value.min()
	lwL2 = net[net.index.get_level_values('fuel')=='LI_SW_CRUDE'].value.mean()
	lwL3 = net[net.index.get_level_values('fuel')=='LI_SW_CRUDE'].value.max()
	lwH1 = net[net.index.get_level_values('fuel')=='HE_SO_CRUDE'].value.min()
	lwH2 = net[net.index.get_level_values('fuel')=='HE_SO_CRUDE'].value.mean()
	lwH3 = net[net.index.get_level_values('fuel')=='HE_SO_CRUDE'].value.max()

	#llws = [0.7*np.sqrt(l) for l in [lwL1, lwL2, lwL3, lwH1, lwH2, lwH3]]
	llws = [0.7*np.sqrt(l) for l in [50, 250, 500, 50, 250, 500]]
	lL1 = Line2D([], [], linewidth=llws[0], color=set2[1], alpha=a) 
	lL2 = Line2D([], [], linewidth=llws[1], color=set2[1], alpha=a) 
	lL3 = Line2D([], [], linewidth=llws[2], color=set2[1], alpha=a)
	lH1 = Line2D([], [], ls='dashed', linewidth=llws[3], color=set2[0], alpha=a)
	lH2 = Line2D([], [], ls='dashed', linewidth=llws[4], color=set2[0], alpha=a)
	lH3 = Line2D([], [], ls='dashed', linewidth=llws[5], color=set2[0], alpha=a)
	 
	# Set three legend labels to be min, mean and max of countries extensions 
	# (rounded up to 10k km2)
	#rnd = 10000 #round(l/rnd)*rnd
	# labelsL = [str(int(l)+1) for l in [lwL1,lwL2,lwL3] ]
	# labelsH = [str(int(l)+1) for l in [lwH1,lwH2,lwH3] ]
	labelsL = [str(int(l)) for l in [50,250,500] ]
	labelsH = [str(int(l)) for l in [50,250,500] ]
	 
	# Position legend in lower right part
	# Set ncol=3 for horizontally expanding legend
	leg2 = ax.legend([lL1, lL2, lL3], labelsL, ncol=3, frameon=False, fontsize=20, 
	                bbox_to_anchor=[.5, 0.12], handlelength=3, 
	                #bbox_to_anchor=[.55, 0.12], handlelength=3, 
	                handletextpad=1, columnspacing=2, title='Light crude (kbpd)')
	leg3 = ax.legend([lH1, lH2, lH3], labelsH, ncol=3, frameon=False, fontsize=20, 
	                bbox_to_anchor=[1, 0.12], handlelength=3, 
	                #bbox_to_anchor=[.75, 0.12], handlelength=3, 
	                handletextpad=1, columnspacing=2, title='Heavy crude (kbpd)')	 
	# Customize legend title
	# Set position to increase space between legend and labels
	plt.setp(leg2.get_title(), fontsize=22, alpha=a)
	leg2.get_title().set_position((0, 0))
	plt.setp(leg3.get_title(), fontsize=22, alpha=a)
	leg3.get_title().set_position((0, 0))
	# Customize transparency for legend labels
	[plt.setp(label, alpha=a) for label in leg2.get_texts()]
	ax.add_artist(leg1)
	ax.add_artist(leg2)
	ax.text(.005, .95, 'Mode: %s, Year: %i' %(mode,yyyy),transform=ax.transAxes,fontsize=22,ha='left' )

	plt.tight_layout()
	#update_text_slopes()
	if save=='y':
		plt.savefig('./' + '%s_%s_%s.pdf' %(scenario,mode,yyyy), format='pdf',bbox='tight',dpi=1200,pad_inches=0)        
	plt.show()
	return net, xlimLU,ylimLU

	# if __name__ == '__main__':
	# 	usmap('base','Pipeline')

os.system("printf '\a'") # or '\7'
