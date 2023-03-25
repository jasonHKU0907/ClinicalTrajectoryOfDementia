

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


dpath1 = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/NA80/'
dpath2 = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/NA80/'
f_df = pd.read_csv(dpath1 + 'S20_Results.csv')
f_df = f_df.loc[f_df.Figure3 == 1]
f_df.reset_index(inplace = True)
f_df.drop('index', axis = 1, inplace = True)
f_lst = ['eid', 'Gender', 'case_control', 'BL2DM_yrs'] + f_df.FieldID_full.tolist()
mydf = pd.read_csv(dpath2 + 'S51_case_control_ukb_pheno_NA80.csv', usecols= f_lst)
mydf.sort_values(by = 'eid', ascending = True, inplace = True)
my_f = mydf.columns.tolist()[4:]

newdf = pd.DataFrame()
mydf_case = mydf.loc[mydf.case_control == 1]
mydf_control = mydf.loc[mydf.case_control == 0]


for f in my_f:
    nb_levels = len(mydf[f].value_counts())
    tmpdf = mydf[f]
    if nb_levels >= 3:
        direction = int(f_df.direction.iloc[f_df.index[f_df.FieldID_full == f]])
        newdf[f] = continuous2binary(mydf_control, mydf, f, direction, alpha = 0.10)
    elif nb_levels == 2:
        newdf[f] = mydf[f]



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


#newdf['1200-0.0'] = mydf['1200-0.0']
#newdf['1200-0.0'] = newdf['1200-0.0'] > 2
#newdf['1200-0.0'] = newdf['1200-0.0'].astype('int')
#newdf['1200-0.0'].iloc[mydf[mydf['1200-0.0'].isnull()].index.tolist()] = np.nan

#newdf['22036-0.0'] = mydf['22036-0.0']
#newdf['22036-0.0'] = newdf['22036-0.0'] == 0
#newdf['22036-0.0'] = newdf['22036-0.0'].astype('int')
#newdf['22036-0.0'].iloc[mydf[mydf['22036-0.0'].isnull()].index.tolist()] = np.nan

newdf = pd.concat((mydf[['eid', 'case_control', 'BL2DM_yrs']], newdf), axis = 1)

newdf = newdf.loc[newdf.BL2DM_yrs<=15]
newdf.reset_index(inplace = True)
newdf.drop('index', axis = 1, inplace = True)
#newdf['BL2DM_YR'] = pd.cut(newdf.BL2DM_yrs, bins=[0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15],
#                           labels = [1.5, 4, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13.5])
#time_ivs = [1.5, 4, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13.5]

newdf['BL2DM_YR'] = pd.cut(newdf.BL2DM_yrs, bins=[0, 5, 10, 15],
                           labels = [2.5, 7.5, 12.5])
time_ivs = [2.5, 7.5, 12.5]

tmpdf_prop = pd.DataFrame({'FieldID_full':my_f})

for tiv in time_ivs:
    tmpdf = newdf.loc[newdf.BL2DM_YR == tiv]
    tmpdf_case = tmpdf.loc[tmpdf.case_control == 1]
    print(tmpdf_case.shape)

for tiv in time_ivs:
    tmpdf = newdf.loc[newdf.BL2DM_YR == tiv]
    tmpdf_case = tmpdf.loc[tmpdf.case_control == 1]
    tmpdf_control = tmpdf.loc[tmpdf.case_control == 0]
    nb_case, nb_control = len(tmpdf_case), len(tmpdf_control)
    case_all_obs = len(tmpdf_case) - tmpdf_case[my_f].isnull().sum(axis = 0).astype(int)
    case_nb = tmpdf_case[my_f].sum(axis = 0).astype(int)
    case_prop = case_nb / case_all_obs
    case_sd = np.sqrt(case_prop*(1-case_prop)/case_all_obs)
    case_lbd = np.round((case_prop - 1.96*case_sd)*100, 1)
    case_ubd = np.round((case_prop + 1.96*case_sd)*100, 1)
    case_nb_obs, case_prop_ci = [], []
    control_all_obs = len(tmpdf_control) - tmpdf_control[my_f].isnull().sum(axis=0).astype(int)
    control_nb = tmpdf_control[my_f].sum(axis=0).astype(int)
    control_prop = control_nb / control_all_obs
    control_sd = np.sqrt(control_prop * (1 - control_prop) / control_all_obs)
    control_lbd = np.round((control_prop - 1.96 * control_sd) * 100, 1)
    control_ubd = np.round((control_prop + 1.96 * control_sd) * 100, 1)
    control_nb_obs, control_prop_ci = [], []
    delta_prop = case_prop - control_prop
    delta_sd = np.sqrt(case_prop*(1-case_prop)/case_all_obs + control_prop * (1 - control_prop) / control_all_obs)
    delta_lbd = np.round((delta_prop - 1.96 * delta_sd) * 100, 1)
    delta_ubd = np.round((delta_prop + 1.96 * delta_sd) * 100, 1)
    delta_prop_ci = []
    for i in range(len(my_f)):
        all_obs1 = case_all_obs[i]
        nb1 = case_nb[i]
        prop1 = np.round(case_prop[i] * 100, 1)
        lbd1 = case_lbd[i]
        ubd1 = case_ubd[i]
        case_nb_obs.append(str(nb1) + '/' + str(all_obs1))
        case_prop_ci.append(str(prop1) + ' [' + str(lbd1) + ', ' + str(ubd1) + ']')
        all_obs2 = control_all_obs[i]
        nb2 = control_nb[i]
        prop2 = np.round(control_prop[i] * 100, 1)
        lbd2 = control_lbd[i]
        ubd2 = control_ubd[i]
        control_nb_obs.append(str(nb2) + '/' + str(all_obs2))
        control_prop_ci.append(str(prop2) + ' [' + str(lbd2) + ', ' + str(ubd2) + ']')
        prop3 = np.round(delta_prop[i] * 100, 1)
        lbd3 = delta_lbd[i]
        ubd3 = delta_ubd[i]
        delta_prop_ci.append(str(prop3) + ' [' + str(lbd3) + ', ' + str(ubd3) + ']')
    tmpdf_prop['case_nb_obs_' + str(tiv)] = pd.DataFrame(case_nb_obs)
    tmpdf_prop['case_prop_ci_' + str(tiv)] = pd.DataFrame(case_prop_ci)
    tmpdf_prop['control_nb_obs_' + str(tiv)] = pd.DataFrame(control_nb_obs)
    tmpdf_prop['control_prop_ci_' + str(tiv)] = pd.DataFrame(control_prop_ci)
    tmpdf_prop['delta_prop_ci_' + str(tiv)] = pd.DataFrame(delta_prop_ci)



tmpdf = newdf
tmpdf_case = tmpdf.loc[tmpdf.case_control == 1]
tmpdf_control = tmpdf.loc[tmpdf.case_control == 0]
nb_case, nb_control = len(tmpdf_case), len(tmpdf_control)
case_all_obs = len(tmpdf_case) - tmpdf_case[my_f].isnull().sum(axis=0).astype(int)
case_nb = tmpdf_case[my_f].sum(axis=0).astype(int)
case_prop = case_nb / case_all_obs
case_sd = np.sqrt(case_prop * (1 - case_prop) / case_all_obs)
case_lbd = np.round((case_prop - 1.96 * case_sd) * 100, 1)
case_ubd = np.round((case_prop + 1.96 * case_sd) * 100, 1)
case_nb_obs, case_prop_ci = [], []
control_all_obs = len(tmpdf_control) - tmpdf_control[my_f].isnull().sum(axis=0).astype(int)
control_nb = tmpdf_control[my_f].sum(axis=0).astype(int)
control_prop = control_nb / control_all_obs
control_sd = np.sqrt(control_prop * (1 - control_prop) / control_all_obs)
control_lbd = np.round((control_prop - 1.96 * control_sd) * 100, 1)
control_ubd = np.round((control_prop + 1.96 * control_sd) * 100, 1)
control_nb_obs, control_prop_ci = [], []
delta_prop = case_prop - control_prop
delta_sd = np.sqrt(case_prop * (1 - case_prop) / case_all_obs + control_prop * (1 - control_prop) / control_all_obs)
delta_lbd = np.round((delta_prop - 1.96 * delta_sd) * 100, 1)
delta_ubd = np.round((delta_prop + 1.96 * delta_sd) * 100, 1)
delta_prop_ci = []
for i in range(len(my_f)):
    all_obs1 = case_all_obs[i]
    nb1 = case_nb[i]
    prop1 = np.round(case_prop[i] * 100, 1)
    lbd1 = case_lbd[i]
    ubd1 = case_ubd[i]
    case_nb_obs.append(str(nb1) + '/' + str(all_obs1))
    case_prop_ci.append(str(prop1) + ' [' + str(lbd1) + ', ' + str(ubd1) + ']')
    all_obs2 = control_all_obs[i]
    nb2 = control_nb[i]
    prop2 = np.round(control_prop[i] * 100, 1)
    lbd2 = control_lbd[i]
    ubd2 = control_ubd[i]
    control_nb_obs.append(str(nb2) + '/' + str(all_obs2))
    control_prop_ci.append(str(prop2) + ' [' + str(lbd2) + ', ' + str(ubd2) + ']')
    prop3 = np.round(delta_prop[i] * 100, 1)
    lbd3 = delta_lbd[i]
    ubd3 = delta_ubd[i]
    delta_prop_ci.append(str(prop3) + ' [' + str(lbd3) + ', ' + str(ubd3) + ']')
tmpdf_prop['case_nb_obs_ALL'] = pd.DataFrame(case_nb_obs)
tmpdf_prop['case_prop_ci_ALL'] = pd.DataFrame(case_prop_ci)
tmpdf_prop['control_nb_obs_ALL'] = pd.DataFrame(control_nb_obs)
tmpdf_prop['control_prop_ci_ALL'] = pd.DataFrame(control_prop_ci)
tmpdf_prop['delta_prop_ci_ALL'] = pd.DataFrame(delta_prop_ci)


tmpdf_prop = pd.merge(f_df, tmpdf_prop, how = 'right', on = ['FieldID_full'])
tmpdf_prop.to_csv(dpath1 + 'ACD-PropTable.csv', index = False)


a = np.array((12, ))