import os.path

import pandas as pd
#import seaborn as sns
#import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, rcParams
from constants import FIGURES_DIR_NAME


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

params = {'mathtext.default': 'regular' }  # Allows tex-style title & labels
plt.rcParams.update(params)


df = pd.read_csv(r'C:\Users\110414\Aitor\Doctorado\Proyectos\OOD_XAI\OOD_XAI\OOD_XAI_MNIST_Rotated.csv',sep=';')

df_grafico = df.loc[df['Mode'] == 'Predicted',['Angle','SSIM','CW-SSIM']]
df_grafico[['SSIM','CW-SSIM']] = df_grafico[['SSIM','CW-SSIM']].stack().str.replace(',','.').unstack().astype('float32')

fig, ax = plt.subplots()

angle   = df_grafico['Angle'].drop(labels=0)
ssim    = df_grafico['SSIM'].drop(labels=0)
cw_ssim = df_grafico['CW-SSIM'].drop(labels=0)

y = np.arange(5,11)*0.1
x = np.arange(len(angle))  # the label locations
width = 0.3  # the width of the bars

rects1 = ax.bar(x - width/2, ssim, width, label='SSIM',color='salmon')
rects2 = ax.bar(x + width/2, cw_ssim, width, label='CW-SSIM',color='cornflowerblue')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUROC Score',fontsize=15)
ax.set_xlabel('Angle',fontsize=15)
#ax.set_title('Scores by group and gender')
ax.set_ybound(0.5,1)
ax.set_xticks(x)
ax.set_xticklabels(map(str,angle),fontsize=15)
ax.tick_params(axis='y', which='major', labelsize=13)
#ax.set_ytickslabels(y,fontsize=13)
ax.legend(fontsize=15)
fig.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR_NAME+'SSIM_vs_CW_SSIM.pdf'),bbox_inches='tight')
