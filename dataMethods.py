'''
Methods for initial data processing in NACOM 
Tables generated are placed in /Input-Parameters
Author: Jimi Oke
Created: Mar 4 2016
Last Mod: Jul 21 2016
'''

import numpy as np
import pandas as pd
pd.options.display.max_rows = 120
pd.options.display.max_columns = 20
pd.options.display.width = None
# pd.options.mode.sim_interactive = True

dirpath = './Primary-Data/'
names = pd.read_excel(dirpath+'SearchList_NodeRegionNames.xlsx',sheetname='Nodes',parse_cols=6)

names = names.drop('Notes',axis=1) # drop unnecessary column
pcnodes = names.dropna() # producer/consumer nodes [no rail terminal nodes]

pro = names[names.Producer==1]
pro = pro.reset_index()
con = names[names.Consumer==1]

def get_crude_prod(year):
	# Obtain and clean USA Crude Oil Production values (EIA)
	df = pd.read_excel(dirpath+'Production/PET_CRD_CRPDN_ADC_MBBLPD_A.xls',sheetname='Data 1',skiprows=2)
	df = df.set_index('Date')
	#df = df/365 # convert from kbbl to kbpd
	ext_text = ' Field Production of Crude Oil (Thousand Barrels per Day)'
	df.columns = df.columns.map(lambda x: str(x)[:-len(ext_text)])
	df = df.transpose()
	df = df[str(year)]
	# Obtain and clean world production values (IEA)
	# Source: http://www.eia.gov/cfapps/ipdbproject/iedindex3.cfm?tid=5&pid=53&aid=1
	# Manually have to enter "Place" as column header for countries
	df1 = pd.read_excel(dirpath+'Production/IEA-World-Crude-Production.xls',sheetname='Data3',skiprows=2)
	df1 = df1.set_index('Place')
	df1 = df1[year]
	df1 = df1.dropna()

	US_prod = df1.ix['United States'] # US total Crude Oil production (IEA)

	# Populate dataframe with node production values
	d2 = pd.DataFrame(columns=['node_name','node_abb','production'])
	d2.node_name = pro.node_name
	d2.node_abb = pro.node_abb
	d2 = d2.set_index('node_name')
	for node in d2.index:
		if df.index.isin([node]).any():
			d2.ix[node].production = df.ix[node][0]
		elif node in ['Mexico']:
			MX_prod = df1.ix['Mexico']
			d2.ix[node].production = MX_prod
		elif node in ['Eastern Canada','Western Canada']: 
			ec_pr = 0.055 #EC production ratio
			CAN_prod = df1.ix['Canada']
			d2.ix['Eastern Canada'].production = ec_pr*CAN_prod
			d2.ix['Western Canada'].production = (1-ec_pr)*CAN_prod
		elif node in ['Rest of World']:
			RW_prod = df1.ix['World'] - (MX_prod + CAN_prod + US_prod)
			d2.ix[node].production = RW_prod

	# Add PADD 5 Offshore to CA and Gulf to TX		
	d2.ix['California'].production = d2.ix['California'].production + \
		df.ix['Federal Offshore PADD 5'][0]
	d2.ix['Texas'].production = d2.ix['Texas'].production + \
		df.ix['Federal Offshore--Gulf of Mexico'][0]		

	d2 = d2.reset_index()
	return d2, CAN_prod, MX_prod, RW_prod, US_prod

def crude_costs():
	df = pd.read_excel(dirpath+'Crude_Costs.xlsx',sheetname='Sheet1',skiprows=10)
	df = df[['Node','Original']] #df[['Cost ($/bbl)','Node']]	
	df = df.set_index('Node')
	return df

def horizon():
	df = pd.read_csv(dirpath+'Crude_Oil_Reserves_2013.csv')
	df = df.set_index('node')
	return df

def li_sw_prod():
	df = pd.read_csv(dirpath+'LI_SW_Production_Ratios.csv')
	df2 = pd.DataFrame(columns = ['node_name','node_abb','li_sw_ratio'])
	df2.node_name = pro.node_name
	df2.node_abb = pro.node_abb
	df2.li_sw_ratio = df2.node_name.apply(lambda x: float(df.LI_SW_prod[df.Node==x]))
	return df2

def production():
	## Data
	prod_output = get_crude_prod(2012)
	prodvals = prod_output[0]
	costs = crude_costs()
	reserves = horizon()
	headers = ['year','supplier','node','fuel','exporter','re_exporter','ref','cap', \
		'inv','hor','lin','qud','gol','loss', \
		'emm_type_1','emm_P_1','emm_type_2','emm_P_2', 'comments'
		]

	df = pd.DataFrame(columns=headers)
	## Scalars
	cap_util = 0.95
	cost_factor = 1.3 #.8
	gol_constant = 3 #3
	marg_cost_factor = .95 #0.7
	min_quad_cost = 0.00002
	yyyy = int(2012)
	loss_factor = 0.015
	fuel_type = "CRUDE"

	def lin_costs(node):
		if node=='EC':
			cost = costs.ix[node].min()
		else:
			cost = costs.ix[node].min()[0]
		cost = cost*cost_factor
		return cost

	def quad_costs(node):
		if node=='EC':
			mcost = costs.ix[node].max()
		else:
			mcost = costs.ix[node].max()[0]		
		mcost = mcost*marg_cost_factor # marginal cost
		lcost = lin_costs(node)
		ref_prod = float(df.ref[df.node==node])
		c1 = 0.5*( mcost-lcost + gol_constant*(np.log(1 - cap_util)) )/ref_prod
		c2 = min_quad_cost
		cost = np.max([c1,c2])
		return cost

	def inv(node):
		ede = {} # exploration and development expenditures
		ede['CAN'] = 8594/prod_output[1]
		#ede['CAN'] = ede['CAN']
		ede['MEX'] = 2880/prod_output[2]
		ede['RW'] = 101041/prod_output[3]
		ede['USA'] = 53386/prod_output[4]
		if node in ['EC','WC']:
			inv_cost = ede['CAN']
		elif node=='MX':
			inv_cost = ede['MEX']
		elif node=='RW':
			inv_cost = ede['RW']
		else:
			inv_cost = ede['USA']
		inv_cost = inv_cost*(1000/365)
		return inv_cost

	df.supplier = pro.node_name
	df.node = pro.node_abb
	df.fuel = fuel_type	
	df.exporter = 1  
	df.re_exporter = 1
	df.ref = prodvals.production
	df.cap = df.ref/cap_util
	df.hor = df.node.apply(lambda x: float(reserves.ix[x].hor))
	df.inv = df.node.apply(inv)
	df.inv = df.inv.apply(int)
	df.lin = df.node.apply(lin_costs)
	df.qud = df.node.apply(quad_costs)
	df.gol = gol_constant
	df.loss = loss_factor
	df.emm_type_1 = "GHG"
	df.emm_P_1 = 0.12
	df.emm_type_2 = ""
	df.emm_P_2 = 0
	df.comments = ""
	df.year = yyyy

	## Transform to LI_SW and HE_SO
	li_sw = li_sw_prod()
	df['li_sw'] = df.node.apply(lambda x: float(li_sw.li_sw_ratio[li_sw.node_abb==x]))

	df2 = df.copy() # create new df for heavy-sour crude
	df.fuel = "LI_SW_"+fuel_type
	df2.fuel = "HE_SO_"+fuel_type

	# change refs and caps accordingly 
	df.ref = df.ref*df.li_sw
	df.cap = df.cap*df.li_sw
	df.hor = df.hor*df.li_sw
	df2.ref = df2.ref*(1-df2.li_sw)
	df2.cap = df2.cap*(1-df2.li_sw)
	df2.hor = df2.hor*(1-df2.li_sw)

	df3 = pd.concat([df,df2]) # combined dataframe

	## Clean up
	# for now assuming same costs for HE_SO and LI_SW by node
	# makes sense as costs were obtained from real data reflecting crude qualities
	df3 = df3.sort('supplier')
	df3 = df3.reset_index(drop=True)
	#df3 = df3[df3.ref!=0] # remove nodes with no HE_SO production
	df3 = df3.drop('li_sw',axis=1) # we don't need the ratios in final result
	df3.supplier = df3.node # finally replace full names with abbrvs as well
	df3.to_csv(dirpath+'Input-Parameters/Production_Nodes.csv',index=False,header=True)
	
	#def marg_cost_check(prod):
	df3['mc_check'] = df3.lin + 2*df3.qud - (np.log(1-cap_util))*df3.gol


	return df3



def get_ref_lisw_yields(year):
	""" 
	Obtain refinery yield ratios for light-sweet crude oil
	"""
	df = pd.read_csv(dirpath+'RefineryInputs/yield-rates-API-averages.csv')
	df = df.set_index('node')
	df = df[str(year)]
	if year==2012:
		df.ix['EC'] = 0.75 #0.6
		df.ix['MX'] = 0.68 # to match production ratio
		df.ix['RW'] = 0.471 #.469 #0.4609995	
		df.ix['WC'] = 0.45 #.65 #changed to match production ratio

	return df


def get_demand(year):
	df = pd.read_csv(dirpath+'US_Crude_Oil_Demand.csv') # in kilo barrels per day
	#df = df.transpose()
	df = df.set_index('year')
	df = df.ix[year]
	df = pd.DataFrame(df)
	df = df.ix[con.node_abb.tolist()]

	if year==2012:
		# refinery utilization rates for CAN, RW, MEX
		EC_util = .666412214 # Source: http://canadianfuels.ca/en/refining-sites-and-capacity
		WC_util = .79 	#.793 #.892763731 # Source: http://canadianfuels.ca/en/refining-sites-and-capacity
		RW_util =  1.012#1.01905 #.7  http://www.bp.com/en/global/corporate/about-bp/energy-economics/statistical-review-of-world-energy/review-by-energy-type/oil/refinery-throughputs.html
		MX_util = .8 #.88	
		# high RW_util makes up for storage not accounted for in model right now

		WC_ref_ratio = 0.6 #how much being refined in W vs E in terms of total demand

		df.ix['EC'] = EC_util*(1-WC_ref_ratio)*1918.455 # Source: IEA
		df.ix['WC'] = WC_util*WC_ref_ratio*1918.455 # Source: IEA
		df.ix['RW'] = RW_util*66809.38 # Source: IEA
		df.ix['MX'] = MX_util*1540 #1740 Source: IEA
	elif year==2014:
		# refinery utilization rates for CAN, RW, MEX
		EC_util = .8974 # Source: http://canadianfuels.ca/en/refining-sites-and-capacity
		WC_util = 1	#(greater than 1; cap maxed!) Source: http://canadianfuels.ca/en/refining-sites-and-capacity
		RW_util = .8#1.012#1.01905 #.7  http://www.bp.com/en/global/corporate/about-bp/energy-economics/statistical-review-of-world-energy/review-by-energy-type/oil/refinery-throughputs.html
		MX_util = .758 # http://www.reuters.com/article/2014/11/25/us-mexico-oil-refineries-idUSKCN0J91YB20141125#weLbBIDvrQTQA3OU.97
		# high RW_util makes up for storage not accounted for in model right now
		WC_ref_ratio = 0.6 #how much being refined in W vs E in terms of total demand
		df.ix['EC'] = EC_util*1247 # Source: http://canadianfuels.ca/en/refining-sites-and-capacity
		df.ix['WC'] = WC_util*WC_ref_ratio*626 # Source: http://canadianfuels.ca/en/refining-sites-and-capacity
		df.ix['RW'] = RW_util*66809.38+3000 # Source: IEA
		df.ix['MX'] = MX_util*1540 #1540 Source: http://www.reuters.com/article/2014/11/25/us-mexico-oil-refineries-idUSKCN0J91YB20141125#weLbBIDvrQTQA3OU.97		
	return df


def consumption():
	## Data
	demand = get_demand(2012)

	headers = ['year','node','sector','fuel','value','elasticity','efficiency', \
		]

	df = pd.DataFrame(columns=headers)

	## Scalars and Strings
	elast = 0.35
	eff = 0.98
	yyyy = int(2012)
	sector = "IND"
	fuel_type = "CRUDE"

	df.node = con.node_abb
	df.sector = sector
	df.fuel = fuel_type	
	df.value = df.node.apply(lambda x: demand.ix[x])
	df.elasticity = elast
	df.efficiency = eff
	df.year = yyyy

	## Transform to LI_SW and HE_SO
	li_sw = get_ref_lisw_yields(2012)
	df['li_sw'] = df.node.apply(lambda x: float(li_sw.ix[x]))
	df2 = df.copy() # create new df for heavy-sour crude
	df.fuel = "LI_SW_"+fuel_type
	df2.fuel = "HE_SO_"+fuel_type

	# change refs and caps accordingly 
	df.value = df.value*df.li_sw # multiply by appr refinery li_sw yield rate
	df2.value = df2.value*(1-df2.li_sw) # multiply by appr refiner he_so yield rate

	df3 = pd.concat([df,df2]) # combine dataframes

	## Clean up
	df3 = df3.sort('node') # sort by node abbrv
	df3 = df3.reset_index(drop=True) 
	df3 = df3[df3.value!=0] # remove nodes with no he_so/li_sw consumption
	df3 = df3.drop('li_sw',axis=1) # we don't need the rates in final result
	#quick and dirty

	df3.to_csv(dirpath+'Input-Parameters/Demand_Nodes.csv',index=False,header=True)

	return df3

def variation(quant):
	"""Constructs Seasonality_Varation_? table for Demand or Production"""
	numyears = 3
	###
	### DEMAND ###
	###
	if quant == "Demand":
		headers = ['year','type','area','season','sector','fuel','quantity','price','elasticity']
		c = consumption()
		con2012 = get_demand(2012)[2012]
		con2014 = get_demand(2014)[2014]
		quant2015 = (con2014/con2012).tolist() #Assume 2014 figures for 2015	
		quant2015 = [val for val in quant2015 for __ in (0,1)] #duplicate each list item (to acc for HE and LI)

		# Generate list of years
		year = 2012
		yListFull = []
		while year <= 2018:
			yList = np.tile(year, len(c)).tolist()
			yListFull = yListFull + yList
			year = year + 3

		typeArray = np.tile("Nodal",len(yListFull)) #for both demand and prod
		seasonArray = np.tile("Y",len(yListFull))		
		areaArray = np.tile(c.node.tolist(), numyears)
		sectorArray = np.tile(c.sector.tolist(), numyears)
		fuelArray = np.tile(c.fuel.tolist(), numyears)

		d = {
			'year' : yListFull,'type' : typeArray,'area' : areaArray,'season' : seasonArray, 
			'sector' : sectorArray, 'fuel' : fuelArray, 'quantity' : 0, 'price' : "", 'elasticity' : ""
		}
		df = pd.DataFrame(d, columns=headers)
		df.quantity[df.year==2012] = 1
		df.quantity[df.year==2015] = quant2015 
		df.quantity[df.year==2018] = [1.07*i for i in quant2015] # assuming same rate for 2015
		df = df.set_index(['year','area','fuel'])
		for i in ['LI_SW_CRUDE','HE_SO_CRUDE']:
			df.loc[(2015,'RW',i),'quantity'] = 1.03
			df.loc[(2018,'RW',i),'quantity'] = 1.07

		# CANADA
		df.loc[(2018,'WC','HE_SO_CRUDE'),'quantity'] = 1.38 # CAPP Report
		df.loc[(2018,'EC','HE_SO_CRUDE'),'quantity'] = 1.38 # CAPP Report
		df.loc[(2018,'WC','LI_SW_CRUDE'),'quantity'] = 1.09 # CAPP Report
		df.loc[(2018,'EC','LI_SW_CRUDE'),'quantity'] = 1.09 # CAPP Report

		# MEXICO
		for i in ['LI_SW_CRUDE','HE_SO_CRUDE']:
			df.loc[(2018, 'MX',i),'quantity'] = .92 #http://www.eia.gov/beta/international/analysis.cfm?iso=MEX
			

		df = df.reset_index()
		df = df[headers]
		df.to_csv(dirpath+'Input-Parameters/Seasonality_Variation_Demand.csv',index=False,header=True)
	###
	### PRODUCTION ###
	###
	elif quant == "Production":
		prod2012 = get_crude_prod(2012)[0].production
		prod2014 = get_crude_prod(2014)[0].production
		quant2015 = (prod2014/prod2012).tolist() #Assume 2014 figures for 2015	
		quant2015 = [val for val in quant2015 for __ in (0,1)] #duplicate each list item (to acc for HE and LI)
		headers = ['year','type','area','season', 'fuel','quantity']
		p = production()
		# Generate list of years
		year = 2012
		yListFull = []
		while year <= 2018:
			yList = np.tile(year, len(p)).tolist()
			yListFull = yListFull + yList
			year = year + 3

		typeArray = np.tile("Nodal",len(yListFull))  
		areaArray = np.tile(p.node.tolist(), numyears)
		seasonArray = np.tile("Y",len(yListFull))		
		fuelArray = np.tile(p.fuel.tolist(), numyears)
		d = {
			'year' : yListFull,'type' : typeArray,'area' : areaArray,'season' : seasonArray, 
			'fuel' : fuelArray, 'quantity' : 0 
		}
		df = pd.DataFrame(d, columns=headers)
		df.quantity[df.year==2012] = 1
		df.quantity[df.year==2015] = quant2015


		df = df.set_index(['year','area','fuel'])
		#find PADD region of area (node):
		n2padd = names.set_index('node_abb')
		# Obtain growth factor projections for 2018 from EIA
		# Source: https://www.eia.gov/analysis/petroleum/crudetypes/pdf/crudetypes.pdf (pp. 8-15)
		for i,j in df.ix[(2018)].index:
			if n2padd.ix[i].Region_name == 'PADD5':  #Southwest region
				if n2padd.ix[i].Region_name == 'AK': #exception for Alaska
					growth = .811
				else:
					growth = 1.57 #2.2/1.4 (p.10)
			elif n2padd.ix[i].Region_name == 'PADD4': #Rockies
				growth = 1.5 #0.6/0.4 (p.11)
			elif n2padd.ix[i].Region_name == 'PADD3': #Gulf
				growth = 2.09 #2.3/1.1 (p.9)
			elif n2padd.ix[i].Region_name == 'PADD2': #Northern Plains/Midcontinent
				if n2padd.ix[i].Region_name in ['NM','OK','KS']: # for Midcontinental 
					growth = 0.8 #.4/.5 (p.13)
				else: # for Northern Great plains
					growth = 2.4 #1.8/.75 (p.12)
			df.loc[(2018,i,j),'quantity'] = growth
		
		# REST OF WORLD
		for i in ['LI_SW_CRUDE','HE_SO_CRUDE']:
			df.loc[(2018,'RW',i),'quantity'] = 1.1 #estimate

		# CANADA growth factors (numbers in cu. meters/day):
		# Sources: https://www.neb-one.gc.ca/nrg/sttstc/crdlndptrlmprdct/stt/archive/stmtdprdctnrchv-eng.html (2012)
		# https://www.neb-one.gc.ca/nrg/sttstc/crdlndptrlmprdct/stt/stmtdprdctn-eng.html (2015)
		df.loc[(2015, 'EC','LI_SW_CRUDE'),'quantity'] = 29082/32206. #0.98 
		df.loc[(2018, 'EC','LI_SW_CRUDE'),'quantity'] = 260/200. #1.3 	

		df.loc[(2015, 'WC','LI_SW_CRUDE'),'quantity'] = 295819/269279. #1.1 
		df.loc[(2015, 'WC','HE_SO_CRUDE'),'quantity'] = 286229/214648. #1.33
		for i in ['LI_SW_CRUDE','HE_SO_CRUDE']:
			df.loc[(2018, 'WC',i),'quantity'] = 4380/3500. 

		# MEXICO
		for i in ['LI_SW_CRUDE','HE_SO_CRUDE']:
			df.loc[(2018, 'MX', i),'quantity'] = 2800/2941. #~.95 (estimate)

		# CLEAN UP
		df = df.reset_index()
		df = df[headers]
		df.to_csv(dirpath+'Input-Parameters/Seasonality_Variation_Production.csv',index=False,header=True)
	else:
		print "You must enter 'Production' or 'Demand' as function argument"
	return df



def proco():
	p = production()
	c = consumption()
	lisw_p = p[p.fuel=='LI_SW_CRUDE'].ref.sum()
	heso_p = p[p.fuel=='HE_SO_CRUDE'].ref.sum()
	lisw_c = c[c.fuel=='LI_SW_CRUDE'].value.sum()
	heso_c = c[c.fuel=='HE_SO_CRUDE'].value.sum()

	print 'Light-Sweet/Heavy-Sour Production: ', lisw_p, heso_p
	print 'Light-Sweet/Heavy-Sour Consumption: ', lisw_c, heso_c
	print 'Total production = ', lisw_p + heso_p
	print 'Total consumption = ', lisw_c + heso_c