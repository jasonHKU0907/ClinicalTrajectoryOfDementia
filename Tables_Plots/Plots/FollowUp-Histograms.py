

import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt


dpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Results/DM_Analysis/Figures/'
mydf0 = pd.read_csv(dpath + 'S0_DM_Target.csv', usecols= ['eid', 'BL2Now_yrs'])
mydf = pd.read_csv(dpath + 'S51_case_control_ukb_pheno_NA80.csv', usecols= ['eid', 'case_control', 'BL2DM_yrs'])
mydf = pd.merge(mydf, mydf0, how = 'left', on = ['eid'])
tmp0 = mydf.loc[mydf.case_control == 0]
tmp1 = mydf.loc[mydf.case_control == 1]
tmp0 = tmp0.loc[tmp0.BL2DM_yrs<=15]
tmp1 = tmp1.loc[tmp1.BL2DM_yrs<=15]

tmp1.BL2DM_yrs.describe()
tmp1.BL2Now_yrs.describe()

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_facecolor("gainsboro")
ax.grid(which='major', alpha=0.5, linewidth=1, color='white')
ax.hist(-tmp1['BL2DM_yrs'], bins=100, density = True, color = 'cornflowerblue', alpha = 0.7)
ax.set_ylim(0, 0.165)
ax.set_yticks([0, 0.03, 0.06, 0.09, 0.12, 0.15])
ax.tick_params(axis = 'y', labelsize=16)
ax.set_xlim(-15, 0)
ax.set_xticks([-15, -12, -9, -6, -3, 0])
ax.set_xticklabels(['-15', '-12', '-9', '-6', '-3', 'Index'])
ax.tick_params(axis='x', labelsize=16)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.vlines(x=-9.5, ymin = 0, ymax = 0.16, colors='darkviolet', ls='--', lw=3.5)
plt.tight_layout()
ax.set_ylabel('Density', fontsize = 20)
ax.set_xlabel('Baseline visits prior to dementia index (years)', fontsize = 20)
fig.tight_layout()
plt.savefig(outpath + 'BL2DM_yrs.png')


fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_facecolor("gainsboro")
ax.grid(which='major', alpha=0.5, linewidth=1, color='white')
ax.hist(tmp1['BL2Now_yrs'], bins=100, density = True, color = 'springgreen', alpha = 0.9)
ax.set_ylim(0, 0.55)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.tick_params(axis = 'y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.vlines(x=13.7, ymin = 0, ymax = 0.55, colors='olive', ls='--', lw=3.5)
plt.tight_layout()
ax.set_ylabel('Density', fontsize = 20)
ax.set_xlabel('Time of follow-up (years)', fontsize = 20)
fig.tight_layout()
plt.savefig(outpath + 'BL2Now_yrs.png')


