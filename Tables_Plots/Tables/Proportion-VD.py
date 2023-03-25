

import glob
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def get_normalization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        tmp_df[col] = (tmp_df[col]-tmp_df[col].mean()) / tmp_df[col].std()
    return tmp_df

def get_binarization(mydf):
    tmp_df = pd.get_dummies(mydf).iloc[:,1]
    return tmp_df

def continuous2binary(mydf_control, mydf, f, direction, alpha = 0.05):
    male_control_df = mydf_control.loc[mydf_control.Gender == 1]
    female_control_df = mydf_control.loc[mydf_control.Gender == 0]
    if direction == 1:
        threshold_male = male_control_df[f].quantile(1 - alpha)
        tmpdf_male = mydf.loc[mydf.Gender == 1][['eid', f]]
        tmpdf_male[f][tmpdf_male[f] > threshold_male] = 999
        tmpdf_male[f][tmpdf_male[f] <= threshold_male] = 0
        tmpdf_male[f] = tmpdf_male[f]/999
        threshold_female = female_control_df[f].quantile(1 - alpha)
        tmpdf_female = mydf.loc[mydf.Gender == 0][['eid', f]]
        tmpdf_female[f][tmpdf_female[f] > threshold_female] = 999
        tmpdf_female[f][tmpdf_female[f] <= threshold_female] = 0
        tmpdf_female[f] = tmpdf_female[f] / 999
    elif direction == 0:
        threshold_male = male_control_df[f].quantile(alpha)
        tmpdf_male = mydf.loc[mydf.Gender == 1][['eid', f]]
        tmpdf_male[f][tmpdf_male[f] < threshold_male] = -999
        tmpdf_male[f][tmpdf_male[f] >= threshold_male] = 0
        tmpdf_male[f] = tmpdf_male[f] / (-999)
        threshold_female = female_control_df[f].quantile(alpha)
        tmpdf_female = mydf.loc[mydf.Gender == 0][['eid', f]]
        tmpdf_female[f][tmpdf_female[f] < threshold_female] = -999
        tmpdf_female[f][tmpdf_female[f] >= threshold_female] = 0
        tmpdf_female[f] = tmpdf_female[f] / (-999)
    tmpdf = pd.concat((tmpdf_male, tmpdf_female), axis=0)
    tmpdf.sort_values(by = 'eid', ascending = True, inplace = True)
    return tmpdf[f]


dpath1 = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/VD_Analysis/'
dpath2 = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/VD_Analysis/Partition_NA80/'
outpath = dpath2
f_df = pd.read_csv(dpath1 + 'S111_Results.csv')
f_df['direction'] = f_df.OR_0to5yrs > 1
f_df['direction'] = f_df.direction.astype('int')

f_lst = ['eid', 'Gender', 'case_control', 'BL2VD_yrs'] + f_df.FieldID_full.tolist()
mydf = pd.read_csv(dpath2 + 'PriorVD_0to15yrs.csv', usecols= f_lst)
mydf.sort_values(by = 'eid', ascending = True, inplace = True)
my_f = mydf.columns.tolist()[4:]

mydf['924-0.0'].replace([0, 1, 2], [1, 0, 0], inplace = True)

newdf = pd.DataFrame()
mydf_case = mydf.loc[mydf.case_control == 1]
mydf_control = mydf.loc[mydf.case_control == 0]


for f in my_f:
    nb_levels = len(mydf[f].value_counts())
    tmpdf = mydf[f]
    if nb_levels >= 6:
        direction = int(f_df.direction.iloc[f_df.index[f_df.FieldID_full == f]])
        newdf[f] = continuous2binary(mydf_control, mydf, f, direction, alpha = 0.10)
    elif nb_levels == 2:
        newdf[f] = mydf[f]
    elif ((nb_levels >2) & (nb_levels <= 5)):
        ref_level = mydf[f].min()
        tmpdf = mydf[f] > ref_level
        tmpdf = tmpdf.astype('int')
        tmpdf.iloc[mydf[mydf[f].isnull()].index.tolist()] = np.nan
        newdf[f] = tmpdf

newdf = pd.concat((mydf[['eid', 'case_control', 'BL2VD_yrs']], newdf), axis = 1)

newdf['BL2VD_YR'] = pd.cut(newdf.BL2VD_yrs, bins=[0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 20],
                           labels = [1.5, 4, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13.5])

time_ivs = [1.5, 4, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13.5]
tmpdf_prop = pd.DataFrame()


for tiv in time_ivs:
    tmpdf = newdf.loc[newdf.BL2VD_YR == tiv]
    tmpdf_case = tmpdf.loc[tmpdf.case_control == 1]
    tmpdf_control = tmpdf.loc[tmpdf.case_control == 0]
    nb_case, nb_control = len(tmpdf_case), len(tmpdf_control)
    tmpdf_prop['case_' + str(tiv)] = tmpdf_case[my_f].sum(axis = 0) / (nb_case - tmpdf_case[my_f].isnull().sum(axis = 0))
    tmpdf_prop['control_' + str(tiv)] = tmpdf_control[my_f].sum(axis = 0) / (nb_control - tmpdf_control[my_f].isnull().sum(axis = 0))

tmpdf_prop.to_csv(dpath1 + 'PropData.csv')

tmpdf_prop = pd.read_csv(dpath1 + 'PropData.csv')
tmpdf_prop = pd.merge(f_df, tmpdf_prop, how = 'right', on = ['FieldID_full'])
tmpdf_prop.to_csv(dpath1 + 'PropData.csv', index = False)


'''
newdf['BL2MD_YR'] = pd.cut(newdf.BL2DM_yrs, bins=[0, 3, 5, 7, 9, 12, 20],
                           labels = [1.5, 4, 6, 8, 10.5, 13.5])

time_ivs = [1.5, 4, 6, 8, 10.5, 13.5]
tmpdf_prop = pd.DataFrame()

for tiv in time_ivs:
    tmpdf = newdf.loc[newdf.BL2MD_YR == tiv]
    tmpdf_case = tmpdf.loc[tmpdf.case_control == 1]
    tmpdf_control = tmpdf.loc[tmpdf.case_control == 0]
    nb_case, nb_control = len(tmpdf_case), len(tmpdf_control)
    tmpdf_prop['case_' + str(tiv)] = tmpdf_case[my_f].sum(axis = 0) / (nb_case - tmpdf_case[my_f].isnull().sum(axis = 0))
    tmpdf_prop['control_' + str(tiv)] = tmpdf_control[my_f].sum(axis = 0) / (nb_control - tmpdf_control[my_f].isnull().sum(axis = 0))

tmpdf_prop.to_csv(dpath1 + 'PropData1.csv')
'''