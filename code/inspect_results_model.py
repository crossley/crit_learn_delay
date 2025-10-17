import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def inspect_class_I():

    print()
    print("Inspecting Class I models...")

    # NOTE: Class I setup
    d1 = pd.read_csv('../output/param_record_class_I_multi.csv')

    d1 = d1.groupby([
        'sigma_perceptual_noise', 'alpha_actor', 'alpha_critic',
        'eta_perceptual_drift', 'delay_sensitive_update'
    ])[['t2c_delay', 't2c_liti', 't2c_siti']].mean().reset_index()


    d1['psp_class'] = 'Other'

    # delay is greater than long iti and short iti
    d1.loc[(d1['t2c_delay'] > d1['t2c_liti']) & (d1['t2c_delay'] > d1['t2c_siti']),
           'psp_class'] = 'Delay Impaired'

    # long iti is greater than delay and short iti
    d1.loc[(d1['t2c_liti'] > d1['t2c_delay']) & (d1['t2c_liti'] > d1['t2c_siti']),
           'psp_class'] = 'Long ITI Impaired'

    # short iti is greater than delay and long iti
    d1.loc[(d1['t2c_siti'] > d1['t2c_delay']) & (d1['t2c_siti'] > d1['t2c_liti']),
           'psp_class'] = 'Short ITI Impaired'

    # make psp_class an ordered category
    d1['psp_class'] = pd.Categorical(d1['psp_class'],
                                     ordered=True,
                                     categories=[
                                         'Delay Impaired', 'Long ITI Impaired',
                                         'Short ITI Impaired', 'Other'
                                     ])

    # report the number of models in each class
    print(d1['psp_class'].value_counts())

    # Remove 'Other' class for plotting
    # d1 = d1[d1['psp_class'] != 'Other']
    # d1['psp_class'] = d1['psp_class'].cat.remove_unused_categories()

    # report the number of models in each class
    print(d1['psp_class'].value_counts())

    # Report the unique values of each parameter
    print("Unique values of sigma_perceptual_noise:", d1['sigma_perceptual_noise'].unique())
    print("Unique values of alpha_actor:", d1['alpha_actor'].unique())
    print("Unique values of alpha_critic:", d1['alpha_critic'].unique())
    print("Unique values of eta_perceptual_drift:", d1['eta_perceptual_drift'].unique())

#    # NOTE: Inspect class I
#    fig, ax = plt.subplots(figsize=(5, 4))
#    sns.histplot(data=d1,
#                 x='t2c_delay',
#                 stat='density',
#                 common_norm=False,
#                 bins=30,
#                 ax=ax)
#    sns.histplot(data=d1,
#                 x='t2c_liti',
#                 stat='density',
#                 common_norm=False,
#                 bins=30,
#                 ax=ax)
#    sns.histplot(data=d1,
#                 x='t2c_siti',
#                 stat='density',
#                 common_norm=False,
#                 bins=30,
#                 ax=ax)
#    plt.show()

    # NOTE: Figure style from first submission
    sns.set_palette("rocket", 4)
    fig, axx = plt.subplots(2, 2, squeeze=False, figsize=(10, 6))

    dp_delay_true = d1[d1["delay_sensitive_update"] == True].copy()
    dp_delay_false = d1[d1["delay_sensitive_update"] == False].copy()

    ax = axx[0, 0]
    sns.countplot(x='psp_class',
                  data=dp_delay_true,
                  hue='psp_class',
                  stat='proportion',
                  legend=False,
                  ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of\nparameter space', fontsize=12)
    ax.set_xticks(np.arange(0, 4, 1))
    labels = [
        tick.get_text().replace('Impaired', '\nImpaired')
        for tick in ax.get_xticklabels()
    ]
    ax.set_xticklabels(labels, fontsize=10)

    ax = axx[0, 1]
    pr = dp_delay_true.copy()
    pr = pr.melt(id_vars='psp_class',
                 value_vars=[
                     'sigma_perceptual_noise', 'alpha_actor', 'alpha_critic',
                     'eta_perceptual_drift'
                 ],
                 var_name='param')
    pr['value'] = pr.groupby(['param'])['value'].transform(lambda x: x / x.max())
    pr['psp_class_2'] = list(zip(pr['psp_class'], pr['param']))
    sns.boxplot(
        x='param',
        y='value',
        hue='psp_class',
        legend=False,
        data=pr,
        ax=ax,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_xticklabels([
        r'$\sigma$', r'$\alpha_{actor}$', r'$\alpha_{critic}$',
        '$\eta_{S}$'
    ])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)


    ax = axx[1, 0]
    sns.countplot(x='psp_class',
                  data=dp_delay_false,
                  hue='psp_class',
                  stat='proportion',
                  legend=False,
                  ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of\nparameter space', fontsize=12)
    ax.set_xticks(np.arange(0, 4, 1))
    labels = [
        tick.get_text().replace('Impaired', '\nImpaired')
        for tick in ax.get_xticklabels()
    ]
    ax.set_xticklabels(labels, fontsize=10)

    ax = axx[1, 1]
    pr = dp_delay_false.copy()
    pr = pr.melt(id_vars='psp_class',
                 value_vars=[
                     'sigma_perceptual_noise', 'alpha_actor', 'alpha_critic',
                     'eta_perceptual_drift'
                 ],
                 var_name='param')
    pr['value'] = pr.groupby(['param'])['value'].transform(lambda x: x / x.max())
    pr['psp_class_2'] = list(zip(pr['psp_class'], pr['param']))
    sns.boxplot(
        x='param',
        y='value',
        hue='psp_class',
        legend=False,
        data=pr,
        ax=ax,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_xticklabels([
        r'$\sigma$', r'$\alpha_{actor}$', r'$\alpha_{critic}$',
        '$\eta_{S}$'
    ])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)

    axx[0, 0].set_title('A: Delay Sensitive Update', fontsize=12)
    axx[0, 1].set_title('B: Delay Sensitive Update', fontsize=12)
    axx[1, 0].set_title('C: No Delay Sensitive Update', fontsize=12)
    axx[1, 1].set_title('D: No Delay Sensitive Update', fontsize=12)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig('../figures/model_new_class_I_no_stored_criterion.pdf')
    plt.close()


def inspect_class_II():

    print()
    print("Inspecting Class II models...")

    # NOTE: Class II setup
    d2 = pd.read_csv('../output/param_record_class_II_multi.csv')

    d2 = d2.groupby([
        'alpha', 'eta_perceptual_drift', 'eta_criterion_drift',
        'delay_sensitive_update'
    ])[['t2c_delay', 't2c_liti', 't2c_siti']].mean().reset_index()

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

    # report the number of models in each class
    print(d2['psp_class'].value_counts())

    # Remove 'Other' class for plotting
    # d2 = d2[d2['psp_class'] != 'Other']

    sns.set_palette("rocket", 4)

    # NOTE: Inspect class I
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(data=d2,
                 x='t2c_delay',
                 stat='density',
                 common_norm=False,
                 bins=30,
                 ax=ax)
    sns.histplot(data=d2,
                 x='t2c_liti',
                 stat='density',
                 common_norm=False,
                 bins=30,
                 ax=ax)
    sns.histplot(data=d2,
                 x='t2c_siti',
                 stat='density',
                 common_norm=False,
                 bins=30,
                 ax=ax)
    plt.show()

    dp_delay_true = d2[d2["delay_sensitive_update"] == True].copy()
    dp_delay_false = d2[d2["delay_sensitive_update"] == False].copy()

    # NOTE: Hack for solving bar width issue
    new_row = pd.DataFrame([{col: pd.NA for col in d2.columns}])
    new_row['psp_class'] = 'Long ITI Impaired'
    dp_delay_true = pd.concat([dp_delay_true, new_row], ignore_index=True)

    new_row = pd.DataFrame([{col: pd.NA for col in d2.columns}])
    new_row['psp_class'] = 'Short ITI Impaired'
    dp_delay_true = pd.concat([dp_delay_true, new_row], ignore_index=True)

    new_row = pd.DataFrame([{col: pd.NA for col in d2.columns}])
    new_row['psp_class'] = 'Other'
    dp_delay_true = pd.concat([dp_delay_true, new_row], ignore_index=True)

    new_row = pd.DataFrame([{col: pd.NA for col in d2.columns}])
    new_row['psp_class'] = 'Delay Impaired'
    dp_delay_false = pd.concat([dp_delay_false, new_row], ignore_index=True)

    new_row = pd.DataFrame([{col: pd.NA for col in d2.columns}])
    new_row['psp_class'] = 'Short ITI Impaired'
    dp_delay_false = pd.concat([dp_delay_false, new_row], ignore_index=True)

    new_row = pd.DataFrame([{col: pd.NA for col in d2.columns}])
    new_row['psp_class'] = 'Other'
    dp_delay_false = pd.concat([dp_delay_false, new_row], ignore_index=True)

    dp_delay_true['psp_class'] = pd.Categorical(dp_delay_true['psp_class'],
                                     ordered=True,
                                     categories=[
                                         'Delay Impaired',
                                         'Long ITI Impaired',
                                         'Short ITI Impaired',
                                         'Other'
                                         ])

    dp_delay_false['psp_class'] = pd.Categorical(dp_delay_false['psp_class'],
                                     ordered=True,
                                     categories=[
                                         'Delay Impaired',
                                         'Long ITI Impaired',
                                         'Short ITI Impaired',
                                         'Other'
                                         ])

    print(dp_delay_true.psp_class.value_counts())
    print(dp_delay_false.psp_class.value_counts())

    # Report the unique values of each parameter
    print("Unique values of alpha:", d2['alpha'].unique())
    print("Unique values of eta_perceptual_drift:", d2['eta_perceptual_drift'].unique())
    print("Unique values of eta_criterion_drift:", d2['eta_criterion_drift'].unique())
    print("Unique values of delay_sensitive_update:", d2['delay_sensitive_update'].unique())

    # NOTE: Figure style from first submission
    fig, axx = plt.subplots(2, 2, squeeze=False, figsize=(10, 6))

    ax = axx[0, 0]
    sns.countplot(x='psp_class',
                  data=dp_delay_true,
                  hue='psp_class',
                  stat='proportion',
                  legend=False,
                  ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of\nparameter space', fontsize=12)
    ax.set_xticks(np.arange(0, 4, 1))
    labels = [
        tick.get_text().replace('Impaired', '\nImpaired')
        for tick in ax.get_xticklabels()
    ]
    ax.set_xticklabels(labels, fontsize=10)

    ax = axx[0, 1]
    pr = dp_delay_true.copy()
    pr = pr.melt(id_vars='psp_class',
                 value_vars=[
                     'alpha', 'eta_perceptual_drift', 'eta_criterion_drift'
                 ],
                 var_name='param')
    pr['value'] = pr.groupby(['param'])['value'].transform(lambda x: x / x.max())
    sns.boxplot(
        x='param',
        y='value',
        hue='psp_class',
        legend=False,
        data=pr,
        ax=ax,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels([
        r'$\alpha$', r'$\eta_{x}$', '$\eta_{c}$'
    ])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)


    ax = axx[1, 0]
    sns.countplot(x='psp_class',
                  data=dp_delay_false,
                  hue='psp_class',
                  stat='proportion',
                  legend=False,
                  ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Proportion of\nparameter space', fontsize=12)
    ax.set_xticks(np.arange(0, 4, 1))
    labels = [
        tick.get_text().replace('Impaired', '\nImpaired')
        for tick in ax.get_xticklabels()
    ]
    ax.set_xticklabels(labels, fontsize=10)

    ax = axx[1, 1]
    pr = dp_delay_false.copy()
    pr = pr.melt(id_vars='psp_class',
                 value_vars=[
                     'alpha', 'eta_perceptual_drift', 'eta_criterion_drift'
                 ],
                 var_name='param')
    pr['value'] = pr.groupby(['param'])['value'].transform(lambda x: x / x.max())
    sns.boxplot(
        x='param',
        y='value',
        hue='psp_class',
        legend=False,
        data=pr,
        ax=ax,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.set_xlabel('')
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels([
        r'$\alpha$', r'$\eta_{x}$', '$\eta_{c}$'
    ])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)

    axx[0, 0].set_title('A: Delay Sensitive Update', fontsize=12)
    axx[0, 1].set_title('B: Delay Sensitive Update', fontsize=12)
    axx[1, 0].set_title('C: No Delay Sensitive Update', fontsize=12)
    axx[1, 1].set_title('D: No Delay Sensitive Update', fontsize=12)

    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig('../figures/model_new_class_II_yes_stored_criterion.pdf')
    plt.close()

if __name__ == "__main__":
    inspect_class_I()
    inspect_class_II()

# # NOTE: old figure
# fig = plt.figure(figsize=(10, 8))
# 
# xywhA = [0.1, 0.6, 0.375, 0.3]
# xywhC = [0.1, 0.15, 0.375, 0.3]
# xywhB = [0.55, 0.55, 0.4, 0.45]
# xywhD = [0.55, 0.1, 0.4, 0.45]
# 
# ax1 = fig.add_axes(xywhA)
# ax2 = fig.add_axes(xywhC)
# ax3 = fig.add_axes(xywhB, projection='3d')
# ax4 = fig.add_axes(xywhD, projection='3d')
# 
# ax_labels = ['A', 'C', 'B', 'D']
# xyA = (0.0, 0.95)
# xyC = (0.0, 0.5)
# xyB = (0.5, 0.95)
# xyD = (0.5, 0.5)
# xy = [xyA, xyC, xyB, xyD]
# for i, ax in enumerate([ax1, ax2, ax3, ax4]):
#     ax.annotate(ax_labels[i],
#                 xy=xy[i],
#                 xycoords='figure fraction',
#                 horizontalalignment='left',
#                 verticalalignment='top',
#                 fontsize=20)
# 
# # Panel A: Proportion of parameter space
# ax = ax1
# sns.countplot(x='psp_class',
#               data=dp,
#               hue='psp_class',
#               stat='proportion',
#               legend=False,
#               ax=ax)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)
# ax.set_xlabel('')
# ax.set_ylabel('Proportion of parameter space', fontsize=12)
# ax.set_xticks(np.arange(0, 4, 1))
# labels = [
#     tick.get_text().replace('Impaired', '\nImpaired')
#     for tick in ax.get_xticklabels()
# ]
# ax.set_xticklabels(labels, fontsize=10)
# 
# # Panel C: parameter values
# ax = ax2
# pr = dp.copy()
# pr = pr.melt(id_vars='psp_class',
#              value_vars=[
#                  'sigma_perceptual_noise', 'alpha_actor', 'alpha_critic',
#                  'eta_perceptual_drift'
#              ],
#              var_name='param')
# pr['value'] = pr.groupby(['param'])['value'].transform(lambda x: x / x.max())
# pr['psp_class_2'] = list(zip(pr['psp_class'], pr['param']))
# sns.boxplot(
#     x='param',
#     y='value',
#     hue='psp_class',
#     data=pr,
#     ax=ax,
# )
# ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.2), ncol=2)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['bottom'].set_linewidth(1.5)
# ax.set_xlabel('')
# ax.set_ylabel('Parameter Value', fontsize=12)
# ax.set_xticks(np.arange(0, 4, 1))
# ax.set_xticklabels([
#     r'$\sigma$', r'$\alpha_{actor}$', r'$\alpha_{critic}$',
#     '$\eta_{perceptual\_drift}$'
# ])
# for tick in ax.get_xticklabels():
#     tick.set_fontsize(10)
# 
# unique_classes = [
#     'Delay Impaired', 'Long ITI Impaired', 'Short ITI Impaired', 'Other'
# ]
# 
# ax = ax3
# for i, cls in enumerate(unique_classes):
#     pr = dp[dp['psp_class'] == cls]
#     color = ['C0', 'C1', 'C2', 'C3'][i]
# 
#     ax.scatter(pr['t2c_delay'],
#                pr['t2c_liti'],
#                pr['t2c_siti'],
#                color=color,
#                marker='o',
#                label=f'Class {cls}',
#                alpha=.5)
# 
# ax.set_xlabel('FB Delay', fontsize=12)
# ax.set_ylabel('Long ITI', fontsize=12)
# ax.set_zlabel('Control', fontsize=12)
# # TODO: decide about axis ranges
# # ax.set_xlim(0, 1)
# # ax.set_ylim(0, 1)
# # ax.set_zlim(0, 1)
# ax.view_init(elev=15, azim=45)
# 
# ax = ax4
# for i, cls in enumerate(unique_classes):
#     print(cls)
#     pr = dp[dp['psp_class'] == cls]
#     color = ['C0', 'C1', 'C2', 'C3'][i]
# 
#     ax.scatter(pr['alpha_actor'],
#                pr['alpha_critic'],
#                pr['sigma_perceptual_noise'],
#                color=color,
#                marker='o',
#                label=f'Class {cls}',
#                alpha=.5)
# 
# ax.set_xlabel(r'$\alpha_{actor}$', fontsize=16)
# ax.set_ylabel(r'$\alpha_{critic}$', fontsize=16)
# ax.set_zlabel(r'$\sigma$', fontsize=16)
# ax.set_xlim(dp['alpha_actor'].min(), dp['alpha_actor'].max())
# ax.set_ylim(dp['alpha_critic'].min(), dp['alpha_critic'].max())
# ax.set_zlim(dp['sigma_perceptual_noise'].min(),
#             dp['sigma_perceptual_noise'].max())
# # TODO: decide about axis ranges
# # ax.set_xticks(np.arange(0, 0.25, 0.05))
# # ax.set_yticks(np.arange(0, 0.25, 0.05))
# # ax.set_zticks(np.arange(0, 10, 1))
# ax.view_init(elev=15, azim=45)
# 
# plt.savefig('../figures/model_new_class_I.pdf')
# plt.show()
# # plt.close()
