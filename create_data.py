import pandas as pd
import numpy as np


data=pd.read_pickle("/home/fintan/tensorflow/chartdeltadone.pkl")

#the variables I'm using to calculate the target
sofafs = ['pfr', 'peep', 'tv', 'peakp', 'cr','dialysis', 'uo', 'bil','plt','map_a', 'map_c', 'norad', 'adr', 'dbut']
sofa=data[data['interventionID'].isin(sofafs)]

losofs = ['pfr','plt','map_a', 'map_c']
hisofs = ['peep', 'tv', 'peakp', 'cr','dialysis', 'uo', 'bil', 'norad', 'adr', 'dbut']

loso = data[data['interventionID'].isin(losofs)]
hiso = data[data['interventionID'].isin(hisofs)]

loso['valueNum'] = loso.apply(lambda x: -x['valueNumber'], axis=1, reduce=True)
loso=loso.drop(columns='valueNumber')

loso.rename(columns={'valueNum': 'valueNumber'}, inplace=True)

sofam = pd.concat([loso,hiso])

sofamax=sofam.groupby(['encounterid','interventionID',(pd.Grouper(key='chartdelta', freq='12H'))]).agg({'valueNumber':["max"]})
#sofa_binnedlast

dfusk = sofamax.unstack('interventionID')

dfusk_l1 = dfusk.xs('valueNumber', axis=1, drop_level=True)
dfusk_l2 = dfusk_l1.xs('max', axis=1, drop_level=True)

dfusk_l2.columns.get_level_values(0).astype(str)

dfusk_flat = dfusk_l2.copy()
dfusk_flat.columns = dfusk_flat.columns.astype(str)
#dfusk_flat
#dfusk_flat['cr']
#dfusk_flat

#sorts and pads the values
sparseSorted = dfusk_flat.sort_values(['encounterid','chartdelta'],ascending=[True,True])

paddedSorted = sparseSorted.copy()
fillForward = ['cr', 'map_a', 'pfr', 'plt', 'bil']
eids = paddedSorted.index.droplevel(1)
eids = eids[~eids.duplicated(keep='first')]
for eid in eids:
    for iid in fillForward:
        paddedSorted.loc[eid][iid] = paddedSorted.loc[eid][iid].fillna(method='pad')


#Making the target : combined metric of organs failing and organs being supported
#Just a sum of various binary parameters

#Is the patient being mechanically ventilated. Tidal volume and Positive end expiratory pressure are ?
#lungs being supported
paddedSorted['ventilated'] = 0
paddedSorted['ventilated'][paddedSorted['tv'].notna() & paddedSorted['peep'].notna()]  = 1

#Is the patient on continueous venous - venous haemo-dialysis?
#kidneys being supported
paddedSorted['cvvhd'] = 0
paddedSorted['cvvhd'][paddedSorted['dialysis'].notna() & (paddedSorted['dialysis'] >0) ] = 1

#Is the patient receiving drugs to make the heart beat harder, or to squeeze the blood vessels and increase blood pressure?
#Heart being supported
paddedSorted['shock'] = 0
paddedSorted['shock'][paddedSorted['dbut'].notna() & (paddedSorted['dbut'] > 0) ] = 1
paddedSorted['shock'][(paddedSorted['norad'].notna() & (paddedSorted['norad'] > 0)) | (paddedSorted['adr'].notna() & (paddedSorted['adr'] > 0))] = 1

#Creatinine is removed from the blood by the kidneys
#Its a blood test and if it gets too high the kidneys are failing
paddedSorted['aki'] = 0
paddedSorted['aki'][paddedSorted['cr'].notna() & (paddedSorted['cr'] >= 170)] = 1

#Biliruben is removed from the blood by the liver
#if it gets too high it means the liver is failing
paddedSorted['liver'] = 0
paddedSorted['liver'][paddedSorted['bil'].notna() & (paddedSorted['bil'] > 32)] = 1

#If the ratio of oxygen in the blood to oxygen in the inspired air is too low, it means the lungs are failing
paddedSorted['oxy'] = 0
paddedSorted['oxy'][paddedSorted['pfr'].notna() & (abs(paddedSorted['pfr']) < 40) ] = 1

#If the bone marrow isnt producing enought platelets, the blood is failing
paddedSorted['marrow'] = 0
paddedSorted['marrow'][paddedSorted['plt'].notna() & (abs(paddedSorted['plt']) < 150)] = 1

#A combined score of organs failing / being supported
paddedSorted['FiCOF'] = 0
paddedSorted['FiCOF'] = paddedSorted['aki'] + paddedSorted['liver'] + paddedSorted['oxy'] + paddedSorted['marrow'] + paddedSorted['shock'] + paddedSorted['ventilated'] + paddedSorted['cvvhd']


#exporting as pkl
paddedSorted['FiCOF'].to_pickle('/home/fintan/tensorflow/targets/FICOFs.pkl')
