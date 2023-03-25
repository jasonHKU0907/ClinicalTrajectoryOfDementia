


import glob
import os
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
pd.options.mode.chained_assignment = None  # default='warn'

def get_normalization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        tmp_df[col] = (tmp_df[col]-tmp_df[col].mean()) / tmp_df[col].std()
    return tmp_df

def get_binarization(mydf):
    tmp_df = pd.get_dummies(mydf).iloc[:,1]
    return tmp_df

def preprocess_df(mydf, target_f):
    tmp_df = mydf.copy()
    tmp_df = pd.get_dummies(data=tmp_df, columns=['Ethnicity'])
    tmp_df.columns = tmp_df.columns.tolist()[:7] + ['Ethnicity_Others', 'Ethnicity_White', 'Ethnicity_Asian', 'Ethnicity_Black']
    # normalize if it is not binarized variable
    if len(tmp_df[target_f].value_counts()) >= 3:
        #tmp_df[target_f] = get_normalization(tmp_df[[target_f]])
        # remove missing values (row manipulation)
        tmp_df.dropna(axis=0, inplace=True)
    elif len(tmp_df[target_f].value_counts()) == 2:
        # remove missing values (row manipulation)
        tmp_df.dropna(axis=0, inplace=True)
        tmp_df[target_f] = get_binarization(tmp_df[target_f])
    rm_cols = [col for col in tmp_df.columns if len(tmp_df[col].value_counts()) <= 1]
    tmp_df.drop(rm_cols, axis = 1, inplace = True)
    return tmp_df

def read_preprocessed_df(file_path):
    mydf = pd.read_csv(file_path)
    mydf[['BL2DM_yrs', 'Age', 'Education', 'TDI']] = get_normalization(mydf[['BL2DM_yrs', 'Age', 'Education', 'TDI']])
    return mydf

def get_summaries(df, f, type):
    stats = df[f].describe()
    my_nb = int(stats.iloc[0])
    if type == 'continuous':
        my_mean = np.round(stats.iloc[1], 1)
        my_se = stats.iloc[2] / np.sqrt(my_nb)
        my_lci = np.round(my_mean - 1.96 * my_se, 1)
        my_uci = np.round(my_mean + 1.96 * my_se, 1)
        my_stats_out = str(my_mean) + ' [' + str(my_lci) + ', ' + str(my_uci) + ']'
    elif type == 'binary':
        my_stat = int(df[f].sum())
        my_mean_prop = np.round(stats.iloc[1]*100, 1)
        my_stats_out = str(my_stat) + ' [' + str(my_mean_prop) +'%]'
    return my_nb, my_stats_out


dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/NA80/Tables/'
out_file = outpath + 'SummaryTable0369121511.csv'

mydf0 = pd.read_csv(dpath + 'S40_UKB_pheno_matched.csv')
info_df = pd.read_csv(dpath + 'S50_case_control_eid_df_matched_NA80.csv')
mydf0 = pd.merge(info_df, mydf0, how = 'left', on = ['eid'])
target_f = '1220-0.0'
ref_f = ['case_control', 'BL2DM_yrs', 'Age', 'Gender', 'Education', 'TDI', 'Ethnicity']
mydf = mydf0[[target_f] + ref_f]
#mydf[target_f].replace([0, 1, 2], [0, 1, 1], inplace = True)
mydf = preprocess_df(mydf, target_f)
mydf.reset_index(inplace = True)
mydf.drop(['index'], axis = 1, inplace = True)
my_type = 'continuous'

start_lst = [0, 0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 0]
end_lst =   [5, 3, 5, 6, 7, 8, 9, 10,11, 12, 15, 15]

all_result_df = pd.DataFrame()


for i in range(len(start_lst)):
    result_df = pd.DataFrame()
    start, end = int(start_lst[i]), int(end_lst[i])
    suffix = str(start) + 'to' + str(end) + 'yrs'
    my_idx = mydf.index[(mydf['BL2DM_yrs'] > start) & (mydf['BL2DM_yrs'] <= end)]
    tmpdf = mydf.copy()
    tmpdf = tmpdf.iloc[my_idx]
    tmpdf.reset_index(inplace=True)
    tmpdf.drop(['index'], axis=1, inplace=True)
    tmpdf[['BL2DM_yrs', 'Age', 'Education', 'TDI']] = get_normalization(tmpdf[['BL2DM_yrs', 'Age', 'Education', 'TDI']])
    case_nb, case_stats = get_summaries(tmpdf.loc[tmpdf.case_control == 1], target_f, my_type)
    control_nb, control_stats = get_summaries(tmpdf.loc[tmpdf.case_control == 0], target_f, my_type)
    Y = tmpdf['case_control']
    X_uni = tmpdf[target_f]
    X_multi = tmpdf[[target_f] + tmpdf.columns.tolist()[2:]]
    mod_uni = sm.Logit(Y, sm.add_constant(X_uni)).fit()
    mod_multi = sm.Logit(Y, sm.add_constant(X_multi)).fit()
    coef_uni, coef_p_uni = np.round(np.exp(mod_uni.params[1]), 2), mod_uni.pvalues[1]
    coef_multi, coef_p_multi = np.round(np.exp(mod_multi.params[1]), 2), mod_multi.pvalues[1]
    ci_uni, ci_multi = mod_uni.conf_int(alpha=0.05), mod_multi.conf_int(alpha=0.05)
    lci_uni, uci_uni = np.round(np.exp(ci_uni.iloc[1, 0]), 2), np.round(np.exp(ci_uni.iloc[1, 1]), 2)
    lci_multi, uci_multi = np.round(np.exp(ci_multi.iloc[1, 0]), 2), np.round(np.exp(ci_multi.iloc[1, 1]), 2)
    or_uni = str(coef_uni) + ' [' + str(lci_uni) + ', ' + str(uci_uni) + ']'
    or_multi = str(coef_multi) + ' [' + str(lci_multi) + ', ' + str(uci_multi) + ']'
    OR_lst = [target_f, case_nb, case_stats, control_nb, control_stats, or_uni, coef_p_uni, or_multi, coef_p_multi]
    result_df = pd.concat((result_df, pd.DataFrame(OR_lst).T), axis=0)
    result_df.columns = ['FieldID_full', 'nb_case' + suffix, 'stat_case' + suffix, 'nb_control' + suffix,
                         'stat_control' + suffix, 'OR_uni_' + suffix, 'p_uni' + suffix, 'OR_multi_' + suffix,
                         'p_multi_' + suffix]
    all_result_df = pd.concat((all_result_df, result_df), axis = 1)




all_result_df.to_csv('/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/NA80/Tables/IPAQ21.csv')

