import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

	# (ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18), 
	# (ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28), 
	# (ax31, ax32, ax33, ax34, ax35, ax36, ax37, ax38), 
	# (ax41, ax42, ax43, ax44, ax45, ax46, ax47, ax48), 
	# (ax51, ax52, ax53, ax54, ax55, ax56, ax57, ax58), 
	# (ax61, ax62, ax63, ax64, ax65, ax66, ax67, ax68), 
	# (ax71, ax72, ax73, ax74, ax75, ax76, ax77, ax78), 
	# (ax81, ax82, ax83, ax84, ax85, ax86, ax87, ax88)

val2012pipe = pd.read_csv('2012pipe.csv', index_col=0)
val2012rail = pd.read_csv('2012rail.csv', index_col=0)
val2012ship = pd.read_csv('2012ship.csv', index_col=0)
ref2012pipe = pd.read_csv('ref2012pipe.csv', index_col=0)
ref2012rail = pd.read_csv('ref2012rail.csv', index_col=0)
ref2012ship = pd.read_csv('ref2012ship.csv', index_col=0)


def cleanStack(data):
	data = data.fillna(0)
	data.index.names = ['region_out']
	data = data.stack()
	data = pd.DataFrame(data)
	data.columns = ['flow']
	data.index = data.index.set_names(['region_out','region_in'])
	return data

val2012pipe = cleanStack(val2012pipe)
val2012rail = cleanStack(val2012rail)
val2012ship = cleanStack(val2012ship)
ref2012pipe = cleanStack(ref2012pipe)
ref2012rail = cleanStack(ref2012rail)
ref2012ship = cleanStack(ref2012ship)

#dflist = [val2012pipe, val2012rail, val2012ship, ref2012pipe, ref2012ship, ref2012rail]

def plotFlowCal(mode):
	if mode=='pipe':
		val = val2012pipe
		ref = ref2012pipe
	elif mode=='rail':	
		val = val2012rail
		ref = ref2012rail
	elif mode=='ship':
		val = val2012ship
		ref = ref2012ship

	f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex='col', sharey='row')
	axlist = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
	regionList = val.index.levels[0].tolist()
	bar_width = 0.2
	opacity = 0.4
	ymax = np.max([val.max(),ref.max()])
	i = 0
	for ax in axlist:
		dfVal = val.ix[(regionList[i])]
		dfRef = ref.ix[(regionList[i])]
		ind = np.arange(len(regionList))
		rects1 = ax.bar(ind, dfVal.flow, width= bar_width, alpha = opacity, color = 'b', label= 'Model',linewidth=0)
		rects2 = ax.bar(ind + bar_width, dfRef.flow, width=bar_width, alpha = opacity, color = 'r', label='Reference',linewidth=0)		
		ax.set_ylabel(regionList[i]) 
		ax.set_ylim([0, ymax])
		#ax.axis('off')
		ax.spines['top'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		if i == len(regionList)-1:
			ax.set_xticks(ind + bar_width)
			ax.set_xticklabels(regionList)
		i += 1
	f.patch.set_visible(False)
	f.subplots_adjust(hspace=0.1,top=.98,bottom=0.02,left=0.03,right=0.51)
	plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
	plt.legend(bbox_to_anchor=[1.2,0.5],handlelength=3, framealpha=0.5,)
	#plt.tight_layout()
	plt.show()
	#return f
