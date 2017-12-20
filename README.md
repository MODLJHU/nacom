# nacom

============================================================================================
R E A D M E
============================================================================================

This directory contains data, methods and code for the North American Crude Oil Model (NACOM)
It also includes the Supplementary Material document to the manuscript:
“Multimodal Flows in Energy Networks with an Application to Crude Oil Markets” (in review)

Olufolajimi Oke
Daniel Huppmann
Max Marshall
Ricky Poulton
Sauleh Siddiqui

July 2016

============================================================================================
Brief Content Description:

<dataMethods.py> shows all the processing required for input into the model
<figureMethods.py> contains code for generating all the figures
<network-map.py> specifically produces Fig.4 in the manuscript

The directory <Base-Year-Validation/> contains the script and files for generating Supplementary
Figures S2, S3 and S4.

Shape files are contained in <cb_2013_us_state_500k>

<primary-data/> is largely self-explanatory. Parameters generated in <dataMethods.py> are placed in 
<Primary-Data/Input-Parameters>

The Excel files <ProdFlows_> contain scenario results used in figureMethods.py for plotting the maps

The model itself (written in GAMS) is not available for publication at this time.


============================================================================================
For further information, contact:

oke@mit.edu, huppmann@iiasa.ac.at, siddiqui@jhu.edu
