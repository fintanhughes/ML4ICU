import pandas as pd
import numpy as np

pd.options.display.max_rows=400
pd.options.display.max_columns=400



data=pd.read_pickle("/home/fintan/tensorflow/chartdeltadone.pkl")

#removes non numeric values for now
dflessrhythm = data[data.interventionID != 'rhythm']
dflessvent = dflessrhythm[dflessrhythm.interventionID != 'vent']
data = dflessvent

#splits day in 4 giving last value
#data_binnedlast=data.groupby(['encounterid','interventionID',(pd.Grouper(key='chartdelta', freq='8H'))]).agg({'valueNumber':['min', 'max']})
data_binnedlast=data.groupby(['encounterid','interventionID',(pd.Grouper(key='chartdelta', freq='6H'))]).agg({'valueNumber':['last']})

#unstacks on iID
dfusk = data_binnedlast.unstack('interventionID')

#flattens horizontal indices
dfusk_l1 = dfusk.xs('valueNumber', axis=1, drop_level=True)
#dfusk.swaplevel('interventionID', '')
dfusk_l2 = dfusk_l1.xs('last', axis=1, drop_level=True)
dfusk_l2.columns.get_level_values(0).astype(str)
dfusk_flat = dfusk_l2.copy()
dfusk_flat.columns = dfusk_flat.columns.astype(str)
dfusk_flat.head()

dfusk_flat.loc[1].index[-1]
### Calculates length of stay and inputs into a new column
for e in dfusk_flat.index.levels[0].unique():
     dfusk_flat.loc[e,'los'] = dfusk_flat.loc[e].index[-1]

# So this works and gives an expanding mean for each pt, and a rolling max
dfusk_flat['temp'].loc[3].expanding().mean()
dfusk_flat['hr'].rolling(2).max().head()

#But once I try iterating this it just spits out NaN in every row.

for i in dfusk_flat.index.levels[0].unique():
     dfusk_flat.loc[i,'exphrmx'] = dfusk_flat['temp'].loc[i].expanding().mean()
#this isnt iterating properly at all.. just giving all NaNs

for i in dfusk_flat.index.levels[0].unique():
     dfusk_flat.loc[i,'exp'] = dfusk_flat['temp'].loc[i].rolling(2).max()
#same problem with rolling

So to make a standard length sequence to give an LSTM I'd like 24/48hr batches of features
    ideally numbered under a new numbered index

#Make 24hr batches? - this isnt the way

daybatch=dfusk_ls.groupby(pd.Grouper(level='chartdelta', freq='24H'))

daybatch.head()
