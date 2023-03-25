

import scipy.stats
import numpy as np
import pandas as pd
from skmisc.loess import loess
import matplotlib.pyplot as plt


dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/Figures/sFigure-2/'
mydf = pd.read_csv(dpath + 'case_control_ukb_pheno_NA80.csv')
mydf = mydf.loc[mydf.BL2DM_yrs <= 15]
mydf.reset_index(inplace=True)

dict_df = pd.read_csv(dpath + 'FigureData.csv')
dict_df = dict_df.loc[dict_df.Figure2 == 1]
dict_df.sort_values(by = 'Feature_Category', inplace = True)
my_f = dict_df.FieldID_full.tolist()
my_f_name = dict_df.Field_rename.tolist()


for i in range(33, len(my_f)):
    target_f = my_f[i]
    target_f_name = my_f_name[i]
    tmpdf = mydf[[target_f, 'BL2DM_yrs', 'case_control']]
    tmpdf.dropna(inplace=True)
    tmpdf.reset_index(inplace=True)
    tmpdf_case = tmpdf.loc[tmpdf.case_control == 1]
    tmpdf_case.reset_index(inplace=True)
    tmpdf_control = tmpdf.loc[tmpdf.case_control == 0]
    tmpdf_control.reset_index(inplace=True)
    x_case = tmpdf_case.BL2DM_yrs
    y_case = tmpdf_case[target_f]
    x_control = tmpdf_control.BL2DM_yrs
    y_control = tmpdf_control[target_f]
    eval_x = pd.DataFrame(np.linspace(0, 15.1, 100)).iloc[:, 0]
    loess_case = loess(x_case, y_case, span=0.75, surface='direct')
    loess_case.fit()
    pred_case = loess_case.predict(eval_x, stderror=True)
    conf_case = pred_case.confidence()
    mean_case = pred_case.values
    lbd_case = conf_case.lower
    ubd_case = conf_case.upper
    loess_control = loess(x_control, y_control, span=0.75, surface='direct')
    loess_control.fit()
    pred_control = loess_control.predict(eval_x, stderror=True)
    conf_control = pred_control.confidence()
    mean_control = pred_control.values
    lbd_control = conf_control.lower
    ubd_control = conf_control.upper
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.set_facecolor("gainsboro")
    ax.grid(which='major', alpha=1, linewidth=1, color='white')
    ax.plot(eval_x, mean_case, color="red", linewidth=3.5)
    ax.fill_between(eval_x, lbd_case, ubd_case, alpha=0.5, lw=0, color="lightsalmon")
    ax.plot(eval_x, mean_control, linestyle='--', color="blue", linewidth=3.5)
    ax.fill_between(eval_x, lbd_control, ubd_control, lw=0, alpha=0.3, color="blue")
    ax.set_xlim([0, 15])
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticks([0, 3, 6, 9, 12, 15])
    ax.set_xticklabels(['Index', '-3', '-6', '-9', '-12', '-15'])
    ax.tick_params(axis='x', labelsize=16)
    ax.invert_xaxis()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    plt.title(target_f_name, size=20)
    plt.tight_layout()
    plt.savefig(dpath + 'Plots/' + target_f_name + '.png')


