from __future__ import print_function

import numpy as np
import pandas as pd

# refiners = pd.read_excel('../SearchList_NodeRegionNames.xlsx',sheetname='Nodes',usecols=[1,6])
# refiners = refiners[refiners.Consumer==1]
# refnodes = refiners.node_abb

yields = pd.read_csv('yield-rates-dummy.csv')

#transf_C(y,n,c,e,f)
# transformation rate by technology c at node n from input e to output f
y = 2012
c = 'ref'
e1 = 'li_sw_crude'
e2 = 'he_so_crude'
f = 'dist'
for i in yields.index:
	row = yields.ix[i]
	n = row.node
	lr = row.lisw
	hr = 1 - row.lisw	
	print('transf_C(\'%i\',\'%s\',\'%s\',\'%s\',\'%s\') = %3.2f; \r' %(y,n,c,e1,f,lr) )
	print('transf_C(\'%i\',\'%s\',\'%s\',\'%s\',\'%s\') = %3.2f; \r' %(y,n,c,e2,f,hr) )

f = '/Users/JimiOke/Dropbox/Oil-On-Trains/Phase-II/MultiMod_v3_2/MultiMod_v3_2/data/mfm_calibration_usa.gms'

