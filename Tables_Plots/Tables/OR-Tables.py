


import glob
import os
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
from statsmodels.stats.multitest import fdrcorrection
pd.options.mode.chained_assignment = None  # default='warn'

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

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
        tmp_df[target_f] = get_normalization(tmp_df[[target_f]])
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

def get_summaries_continuous(df, f):
    stats = df[f].describe()
    my_nb = int(stats.iloc[0])
    my_mean = np.round(stats.iloc[1], 2)
    my_se = np.round(stats.iloc[2] / np.sqrt(my_nb), 3)
    my_lci = np.round(my_mean - 1.96*my_se, 3)
    my_uci = np.round(my_mean + 1.96*my_se, 3)
    my_stats_out = str(my_mean) + ' [' + str(my_lci) + ', ' + str(my_uci) + ']'
    return my_nb, my_stats_out

def get_summaries(df, f, type):
    stats = df[f].describe()
    my_nb = int(stats.iloc[0])
    if type == 'continuous':
        my_mean = np.round(stats.iloc[1], 2)
        my_se = stats.iloc[2] / np.sqrt(my_nb)
        my_lci = np.round(my_mean - 1.96 * my_se, 2)
        my_uci = np.round(my_mean + 1.96 * my_se, 2)
        my_stats_out = str(my_mean) + ' [' + str(my_lci) + ', ' + str(my_uci) + ']'
    elif type == 'binary':
        my_mean = stats.iloc[1]
        my_se = my_mean*(1-my_mean) / np.sqrt(my_nb)
        my_lci = np.round((my_mean - 1.96 * my_se)*100, 2)
        my_uci = np.round((my_mean + 1.96 * my_se)*100, 2)
        my_mean_prop = np.round(my_mean*100, 2)
        my_stats_out = str(my_mean_prop) + ' [' + str(my_lci) + ', ' + str(my_uci) + ']'
    return my_nb, my_stats_out


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


dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/NA80/Tables/'
f_df = pd.read_csv('/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/PhenoData/Feature_Dict.csv')
f_category = pd.read_csv('/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/PhenoData/feat_catorgy.csv')
f_category = f_category[['FieldID_full', 'Feature_Category']]

out_file = outpath + 'S1_Results-035678910111215.csv'
my_files = sorted(glob.glob(dpath + '*.csv'), key=numericalSort)
my_files = ['/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to5yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to3yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_3to5yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_5to6yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_6to7yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_7to8yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_8to9yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_9to10yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_10to11yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_11to12yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_12to15yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to15yrs.csv']

tmp = read_preprocessed_df(my_files[0])
all_result_df = pd.DataFrame({'FieldID_full':tmp.columns[8:]})


for file in my_files:
    mydf = read_preprocessed_df(file)
    suffix = os.path.basename(file).split('_')[1].split('.')[0]
    print((suffix, mydf.shape[0]/11))

for file in my_files:
    mydf = read_preprocessed_df(file)
    suffix = os.path.basename(file).split('_')[1].split('.')[0]
    my_f = mydf.columns.tolist()[8:]
    result_df = pd.DataFrame()
    for f in my_f:
        try:
            tmp_df = mydf[[f] + ['case_control', 'BL2DM_yrs', 'Age', 'Gender', 'Education', 'TDI', 'Ethnicity']]
            if len(tmp_df[f].value_counts()) >= 3:
                tmp_type = 'continuous'
            elif len(tmp_df[f].value_counts()) == 2:
                tmp_type = 'binary'
            case_nb, case_stats = get_summaries(tmp_df.loc[tmp_df.case_control == 1], f, tmp_type)
            control_nb, control_stats = get_summaries(tmp_df.loc[tmp_df.case_control == 0], f, tmp_type)
            tmp_df = preprocess_df(tmp_df, target_f = f)
            Y = tmp_df['case_control']
            X_uni = tmp_df[f]
            X_multi = tmp_df[[f] + tmp_df.columns.tolist()[2:]]
            mod_uni = sm.Logit(Y, sm.add_constant(X_uni)).fit()
            mod_multi = sm.Logit(Y, sm.add_constant(X_multi)).fit()
            coef_uni, coef_p_uni = np.round(np.exp(mod_uni.params[1]),2), mod_uni.pvalues[1]
            coef_multi, coef_p_multi = np.round(np.exp(mod_multi.params[1]),2), mod_multi.pvalues[1]
            ci_uni, ci_multi = mod_uni.conf_int(alpha=0.05), mod_multi.conf_int(alpha=0.05)
            lci_uni, uci_uni = np.round(np.exp(ci_uni.iloc[1, 0]),2), np.round(np.exp(ci_uni.iloc[1, 1]),2)
            lci_multi, uci_multi = np.round(np.exp(ci_multi.iloc[1, 0]),2), np.round(np.exp(ci_multi.iloc[1, 1]),2)
            or_uni = str(coef_uni) + ' [' + str(lci_uni) + ', ' + str(uci_uni) + ']'
            or_multi = str(coef_multi) + ' [' + str(lci_multi) + ', ' + str(uci_multi) + ']'
            OR_lst = [f, case_nb, case_stats, control_nb, control_stats, or_uni, coef_p_uni, or_multi, coef_p_multi]
            result_df = pd.concat((result_df, pd.DataFrame(OR_lst).T), axis = 0)
        except:
            pass
    result_df.columns = ['FieldID_full', 'nb_case' + suffix, 'stat_case' + suffix, 'nb_control' + suffix, 'stat_control' + suffix,
                         'OR_uni_'+ suffix, 'p_uni'+ suffix, 'OR_multi_'+ suffix, 'p_multi_'+ suffix]
    reject_case, p_fdr = fdrcorrection(result_df['p_multi_'+ suffix].fillna(1))
    result_df['p_multi_fdr_'+ suffix] = p_fdr
    all_result_df = pd.merge(all_result_df, result_df, how = 'left', on = ['FieldID_full'])
    print('done' + os.path.basename(file))


all_result_df['FieldID'] = pd.DataFrame([int(ele.split('-')[0]) for ele in all_result_df['FieldID_full'][:398]])
all_result_df = pd.merge(all_result_df, f_category, how = 'left', on = ['FieldID_full'])
all_result_df = pd.merge(all_result_df, f_df, how = 'left', on = ['FieldID'])

all_result_df.to_csv(out_file, index=False)

ref_dict = pd.read_csv(outpath + 'DictTable.csv')
ref_dict = ref_dict
all_result_df1 = pd.merge(all_result_df, ref_dict, how = 'left', on = ['FieldID_full'])
all_result_df1.to_csv(out_file, index=False)

