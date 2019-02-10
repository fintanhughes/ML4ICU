import pandas as pd
import numpy as np

data = pd.read_pickle("/home/fintan/tensorflow/chartdeltadone.pkl")

# removing non numeric features for now
dflessrhythm = data[data.interventionID != 'rhythm']
dflessvent = dflessrhythm[dflessrhythm.interventionID != 'vent']
data = dflessvent

# Put into time bins
# data_binnedlast=data.groupby(['encounterid','interventionID',(pd.Grouper(key='chartdelta', freq='8H'))]).agg({'valueNumber':['min', 'max']})
data_binnedlast=data.groupby(['encounterid','interventionID',(pd.Grouper(key='chartdelta', freq='12H'))]).agg({'valueNumber':['max']})

# unstacking to give unique column for each feature
dfusk = data_binnedlast.unstack('interventionID')

# cleans up the indexes
dfusk_l1 = dfusk.xs('valueNumber', axis=1, drop_level=True)
dfusk_l2 = dfusk_l1.xs('last', axis=1, drop_level=True)
dfusk_l2.columns.get_level_values(0).astype(str)
dfusk_flat = dfusk_l2.copy()
dfusk_flat.columns = dfusk_flat.columns.astype(str)

# THIS calculates length of stay and inputs into a new column
for e in dfusk_flat.index.levels[0].unique():
     dfusk_flat.loc[e,'los'] = dfusk_flat.loc[e].index[-1]

# THIS reorganises column index levels after the groupby.agg
#dfld = dfusk.swaplevel(i=-2, j=-1, axis=1)
#dfld.head()

#[[[[[[ THIS SECTION IS NOT WORKING YET

# attemps at rolling expanding means.. they dont iterate across individual patients
for i in dfusk_flat.index.levels[0].unique():
     dfusk_flat.loc[i,'exphrmx'] = dfusk_flat['temp'].loc[i].expanding().mean()
# this isnt iterating properly at all.. just giving all NaNs

# works for individual patients
#dfusk_ls['hr'].expanding().max()
dfusk_ls['hr'].loc[2].expanding().mean()
dfusk_ls['hr'].rolling(2).max()

for i in dfusk_flat.index.levels[0].unique():
     dfusk_flat.loc[i,'exp'] = dfusk_flat['temp'].loc[i].rolling(2).max()
# this isnt iterating properly at all.. just giving all NaNs

dfusk_flat['temp'].loc[3].expanding().mean()

                                              #]]]]]]]]]

## takes only those with a LOS >= 2 days
tee = pd.to_timedelta('2 days 00:00:00.00000')
dfusk_ls = dfusk_flat[dfusk_flat["los"] >= tee]

# a simplified dataframe of the key features to start with
simft2 = sparseSorted[['hr','map_a', 'map_c', 'temp', 'rr', 'lact', 'wcc', 'ph', 'cr', 'pfr']]

# Pads forwards missing values
paddedSorted = simft2.copy()
fillForward = ['hr','map_a', 'map_c', 'temp', 'rr', 'lact', 'wcc', 'ph', 'cr', 'pfr']
eids = paddedSorted.index.droplevel(1)
eids = eids[~eids.duplicated(keep='first')]
for eid in eids:
    for iid in fillForward:
        paddedSorted.loc[eid][iid] = paddedSorted.loc[eid][iid].fillna(method='pad')
        
        
# exporting out the .pkl
simps=paddedSorted
simps.to_pickle('/home/fintan/tensorflow/features/simps.pkl')
