import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d1 = pd.read_csv('../output/param_record_class_I_multi.csv')
d2 = pd.read_csv('../output/param_record_class_II_multi.csv')

d1 = d1.groupby([
    'sigma_perceptual_noise', 'alpha_actor', 'alpha_critic',
    'eta_perceptual_drift', 'delay_sensitive_update'
])[['t2c_delay', 't2c_liti', 't2c_siti']].mean().reset_index()

d2 = d2.groupby([
    'alpha', 'eta_perceptual_drift', 'eta_criterion_drift',
    'delay_sensitive_update'
])[['t2c_delay', 't2c_liti', 't2c_siti']].mean().reset_index()

d1['psp_class'] = 'Other'

# delay is greater than long iti and short iti
d1.loc[(d1['t2c_delay'] > d1['t2c_liti']) &
       (d1['t2c_delay'] > d1['t2c_siti']),
       'psp_class'] = 'Delay Impaired'

# long iti is greater than delay and short iti
d1.loc[(d1['t2c_liti'] > d1['t2c_delay']) &
       (d1['t2c_liti'] > d1['t2c_siti']),
       'psp_class'] = 'Long ITI Impaired'

# short iti is greater than delay and long iti
d1.loc[(d1['t2c_siti'] > d1['t2c_delay']) &
       (d1['t2c_siti'] > d1['t2c_liti']),
       'psp_class'] = 'Short ITI Impaired'

# make psp_class an ordered category
d1['psp_class'] = pd.Categorical(d1['psp_class'],
                                 ordered=True,
                                 categories=[
                                     'Delay Impaired',
                                     'Long ITI Impaired',
                                     'Short ITI Impaired',
                                     'Other'])

d2['psp_class'] = 'Other'

# delay is greater than long iti and short iti
d2.loc[(d2['t2c_delay'] > d2['t2c_liti']) &
       (d2['t2c_delay'] > d2['t2c_siti']),
       'psp_class'] = 'Delay Impaired'

# long iti is greater than delay and short iti
d2.loc[(d2['t2c_liti'] > d2['t2c_delay']) &
       (d2['t2c_liti'] > d2['t2c_siti']),
       'psp_class'] = 'Long ITI Impaired'

# short iti is greater than delay and long iti
d2.loc[(d2['t2c_siti'] > d2['t2c_delay']) &
       (d2['t2c_siti'] > d2['t2c_liti']),
       'psp_class'] = 'Short ITI Impaired'

# make psp_class an ordered category
d2['psp_class'] = pd.Categorical(d2['psp_class'],
                                 ordered=True,
                                 categories=[
                                     'Delay Impaired',
                                     'Long ITI Impaired',
                                     'Short ITI Impaired',
                                     'Other'])

# TODO: investigate the 'Other' class
# TODO: alpha_critic doesn't need to span [0, 1]
# TODO: both alphas can span [0, 0.2]

# TODO: decide about delay_sensitive_update and eta_perceptual_drift
d1 = d1.loc[d1['delay_sensitive_update'] == True]
d1 = d1.loc[d1['eta_perceptual_drift'] == 0.0]

# NOTE: Figure style from first submission
sns.set_palette("rocket", 4)

fig = plt.figure(figsize=(10, 8))

xywhA = [0.1, 0.6, 0.375, 0.3]
xywhC = [0.1, 0.15, 0.375, 0.3]
xywhB = [0.55, 0.55, 0.4, 0.45]
xywhD = [0.55, 0.1, 0.4, 0.45]

ax1 = fig.add_axes(xywhA)
ax2 = fig.add_axes(xywhC)
ax3 = fig.add_axes(xywhB, projection='3d')
ax4 = fig.add_axes(xywhD, projection='3d')

ax_labels = ['A', 'C', 'B', 'D']
xyA = (0.0, 0.95)
xyC = (0.0, 0.5)
xyB = (0.5, 0.95)
xyD = (0.5, 0.5)
xy = [xyA, xyC, xyB, xyD]
for i, ax in enumerate([ax1, ax2, ax3, ax4]):
    ax.annotate(ax_labels[i],
                xy=xy[i],
                xycoords='figure fraction',
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=20)

ax = ax1
sns.countplot(x='psp_class',
              data=d1,
              hue='psp_class',
              stat='proportion',
              legend=False,
              ax=ax)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.set_xlabel('')
ax.set_ylabel('Proportion of parameter space', fontsize=12)
ax.set_xticks(np.arange(0, 4, 1))
labels = [
    tick.get_text().replace('Impaired', '\nImpaired')
    for tick in ax.get_xticklabels()
]
ax.set_xticklabels(labels, fontsize=10)

ax = ax2
pr = d1.copy()
pr = pr.melt(id_vars='psp_class',
             value_vars=['sigma_perceptual_noise', 'alpha_actor', 'alpha_critic'],
             var_name='param')
pr['value'] = pr.groupby(['param' ])['value'].transform(lambda x: x / x.max())
pr['psp_class_2'] = list(zip(pr['psp_class'], pr['param']))
sns.boxplot(
    x='param',
    y='value',
    hue='psp_class',
    data=pr,
    ax=ax,
)
ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.2), ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.set_xlabel('')
ax.set_ylabel('Parameter Value', fontsize=12)
ax.set_xticks(np.arange(0, 3, 1))
ax.set_xticklabels([r'$\sigma$', r'$\alpha_{actor}$', r'$\alpha_{critic}$'])
for tick in ax.get_xticklabels():
    tick.set_fontsize(10)

unique_classes = d1['psp_class'].unique()

ax = ax3
for i, cls in enumerate(unique_classes):
    pr = d1[d1['psp_class'] == cls]
    color = ['C0', 'C1', 'C2', 'C3'][i]

    ax.scatter(pr['t2c_delay'],
               pr['t2c_liti'],
               pr['t2c_siti'],
               color=color,
               marker='o',
               label=f'Class {cls}',
               alpha=.5)

ax.set_xlabel('FB Delay', fontsize=12)
ax.set_ylabel('Long ITI', fontsize=12)
ax.set_zlabel('Control', fontsize=12)
# TODO: decide about axis ranges
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)
# ax.set_zlim(0, 1)
ax.view_init(elev=15, azim=45)

ax = ax4
for i, cls in enumerate(unique_classes):
    pr = d1[d1['psp_class'] == cls]
    color = ['C0', 'C1', 'C2', 'C3'][i]

    ax.scatter(pr['alpha_actor'],
               pr['alpha_critic'],
               pr['sigma_perceptual_noise'],
               color=color,
               marker='o',
               label=f'Class {cls}',
               alpha=.5)

ax.set_xlabel(r'$\alpha_{actor}$', fontsize=16)
ax.set_ylabel(r'$\alpha_{critic}$', fontsize=16)
ax.set_zlabel(r'$\sigma$', fontsize=16)
ax.set_xlim(d1['alpha_actor'].min(),
            d1['alpha_actor'].max())
ax.set_ylim(d1['alpha_critic'].min(),
            d1['alpha_critic'].max())
ax.set_zlim(d1['sigma_perceptual_noise'].min(),
            d1['sigma_perceptual_noise'].max())
# TODO: decide about axis ranges
# ax.set_xticks(np.arange(0, 0.25, 0.05))
# ax.set_yticks(np.arange(0, 0.25, 0.05))
# ax.set_zticks(np.arange(0, 10, 1))
ax.view_init(elev=15, azim=45)

plt.savefig('../figures/model_new_class_I.pdf')
plt.close()
