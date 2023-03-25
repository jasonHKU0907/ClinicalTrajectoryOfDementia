


import glob
import os
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
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


def get_summaries1(df, f, type):
    stats = df[f].describe()
    my_nb = int(stats.iloc[0])
    if type == 'continuous':
        my_median = np.round(stats.iloc[5], 1)
        my_lci = np.round(stats.iloc[4], 1)
        my_uci = np.round(stats.iloc[6], 1)
        my_stats_out = str(my_median) + ' [' + str(my_lci) + ', ' + str(my_uci) + ']'
    elif type == 'binary':
        my_stat = int(df[f].sum())
        my_mean_prop = np.round(stats.iloc[1]*100, 1)
        my_stats_out = str(my_stat) + ' [' + str(my_mean_prop) +'%]'
    return my_nb, my_stats_out




#dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/'
#dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/AD_Analysis/Partition_NA80/'
dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/VD_Analysis/Partition_NA80/'
#outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/NA80/Tables/'
#outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/AD_Analysis/Tables/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/VD_Analysis/Tables/'
f_df = pd.read_csv('/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/PhenoData/Feature_Dict.csv')
f_category = pd.read_csv('/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/PhenoData/feat_catorgy.csv')
f_category = f_category[['FieldID_full', 'Feature_Category']]

out_file = outpath + 'MainTable1.csv'

my_files = sorted(glob.glob(dpath + '*.csv'), key=numericalSort)
my_files = ['/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to3yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_3to6yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_6to9yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_9to12yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_12to15yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to15yrs.csv']

my_files = ['/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to5yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_5to10yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_10to15yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/Partition_NA80/PriorDM_0to15yrs.csv']

my_files = ['/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/AD_Analysis/Partition_NA80/PriorAD_0to5yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/AD_Analysis/Partition_NA80/PriorAD_5to10yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/AD_Analysis/Partition_NA80/PriorAD_10to15yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/AD_Analysis/Partition_NA80/PriorAD_0to15yrs.csv']


my_files = ['/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/VD_Analysis/Partition_NA80/PriorVD_0to5yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/VD_Analysis/Partition_NA80/PriorVD_5to10yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/VD_Analysis/Partition_NA80/PriorVD_10to15yrs.csv',
            '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/VD_Analysis/Partition_NA80/PriorVD_0to15yrs.csv']

all_result_df = pd.DataFrame()


for file in my_files:
    mydf = pd.read_csv(file)
    suffix = os.path.basename(file).split('_')[1].split('.')[0]
    mydf = mydf.iloc[:,:8]
    mydf = pd.get_dummies(data=mydf, columns=['Ethnicity'])
    mydf.columns = mydf.columns.tolist()[:7] + ['Ethnicity_Others', 'Ethnicity_White', 'Ethnicity_Asian', 'Ethnicity_Black']
    mydf['Education'].replace([2, 3, 4, 5, 7, 8], ['0_10', '10_15', '10_15', '10_15', '15_20', '15_20'], inplace=True)
    mydf = pd.get_dummies(data=mydf, columns=['Education'])
    case_df = mydf.loc[mydf.case_control == 1].iloc[:,3:]
    control_df = mydf.loc[mydf.case_control == 0].iloc[:,3:]
    result_df = pd.DataFrame()
    my_f = case_df.columns.tolist()
    for f in my_f:
        if f in ['Age', 'TDI']:
            type = 'continuous'
        else:
            type = 'binary'
        case_nb, case_stats = get_summaries(case_df, f, type)
        control_nb, control_stats = get_summaries(control_df, f, type)
        output_lst = [f, case_nb, case_stats, control_nb, control_stats]
        result_df = pd.concat((result_df, pd.DataFrame(output_lst).T), axis=0)
    result_df.columns = ['FieldID_full', 'nb_case' + suffix, 'stat_case' + suffix, 'nb_control' + suffix, 'stat_control' + suffix]
    all_result_df = pd.concat((all_result_df, result_df), axis=1)
    print('done' + os.path.basename(file))

all_result_df.to_csv(out_file, index=False)

