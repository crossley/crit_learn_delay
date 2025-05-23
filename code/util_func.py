from imports import *


def load_data_exp_3():
    dir_delay = '../data/delay'
    dir_immed = '../data/immed'
    dir_maip = '../data/maip'
    dir_dual = '../data/dual'

    filenames_delay = [
        os.path.join(dir_delay, f) for f in os.listdir(dir_delay)
    ]
    filenames_immed = [
        os.path.join(dir_immed, f) for f in os.listdir(dir_immed)
    ]
    filenames_maip = [os.path.join(dir_maip, f) for f in os.listdir(dir_maip)]
    filenames_dual = [os.path.join(dir_dual, f) for f in os.listdir(dir_dual)]

    col_names = ['t', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt']

    d_delay = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_delay
    ]
    d_immed = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_immed
    ]
    d_maip = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_maip
    ]

    col_names_dual = [
        't', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt', 'V8', 'V9', 'V10',
        'V11', 'V12', 'rt2'
    ]
    d_dual = [
        pd.read_csv(f, sep='\t', header=None, names=col_names_dual)
        for f in filenames_dual
    ]

    for idx, df in enumerate(d_delay):
        df['sub'] = idx + 101
    for idx, df in enumerate(d_immed):
        df['sub'] = idx + 201
    for idx, df in enumerate(d_maip):
        df['sub'] = idx + 301
    for idx, df in enumerate(d_dual):
        df['sub'] = idx + 401

    d_delay = pd.concat(d_delay, ignore_index=True)
    d_immed = pd.concat(d_immed, ignore_index=True)
    d_maip = pd.concat(d_maip, ignore_index=True)
    d_dual = pd.concat(d_dual, ignore_index=True)

    d_dual = d_dual[['t', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt', 'sub']]

    d_delay['cnd'] = 'Delay'
    d_immed['cnd'] = 'Long ITI'
    d_maip['cnd'] = 'Short ITI'
    d_dual['cnd'] = 'Dual'

    d = pd.concat([d_delay, d_immed, d_maip, d_dual], ignore_index=True)

    d['exp'] = 'Exp 3'

    d = d[['exp', 'sub', 'cnd', 't', 't_prob', 'rt']]

    d['experiment_duration'] = d.groupby(['cnd', 'sub'])['rt'].transform('sum')
    d['experiment_duration'] = d['experiment_duration'] * 5 / 60
    d['trials_completed'] = d.groupby(['cnd', 'sub'])['t'].transform('max')

    d['prob_num'] = d.groupby(['cnd', 'sub'])['t_prob'].transform(lambda x: (x < x.shift(1)).cumsum() + 1)
    d['t2c'] = d.groupby(['cnd', 'sub', 'prob_num'])['t_prob'].transform('count')
    d['nps'] = d.groupby(['cnd', 'sub'])['prob_num'].transform('max')

    return d


def load_data():
    d1 = load_data_exp_1()
    d2 = load_data_exp_2()

    d1['exp'] = 'Exp 1'
    d2['exp'] = 'Exp 2'

    d1 = d1[['exp', 'sub', 'cnd', 't', 't_prob', 'rt']]
    d2 = d2[['exp', 'sub', 'cnd', 't', 't_prob', 'rt']]

    d = pd.concat([d1, d2], ignore_index=True)

    d['experiment_duration'] = d.groupby(['cnd', 'sub'])['rt'].transform('sum')
    d['experiment_duration'] = d['experiment_duration'] * 5 / 60
    d['trials_completed'] = d.groupby(['cnd', 'sub'])['t'].transform('max')

    d['prob_num'] = d.groupby(
        ['cnd',
         'sub'])['t_prob'].transform(lambda x: (x < x.shift(1)).cumsum() + 1)
    d['t2c'] = d.groupby(['cnd', 'sub',
                          'prob_num'])['t_prob'].transform('count')
    d['nps'] = d.groupby(['cnd', 'sub'])['prob_num'].transform('max')

    return d


def load_data_exp_1():

    dir_delay = '../data/delay'
    dir_immed = '../data/immed'
    dir_maip = '../data/maip'

    filenames_delay = [
        os.path.join(dir_delay, f) for f in os.listdir(dir_delay)
    ]
    filenames_immed = [
        os.path.join(dir_immed, f) for f in os.listdir(dir_immed)
    ]
    filenames_maip = [os.path.join(dir_maip, f) for f in os.listdir(dir_maip)]

    col_names = ['t', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt']

    d_delay = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_delay
    ]
    d_immed = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_immed
    ]
    d_maip = [
        pd.read_csv(f, sep='\t', header=None, names=col_names)
        for f in filenames_maip
    ]

    for idx, df in enumerate(d_delay):
        df['sub'] = idx + 101
    for idx, df in enumerate(d_immed):
        df['sub'] = idx + 201
    for idx, df in enumerate(d_maip):
        df['sub'] = idx + 301

    d_delay = pd.concat(d_delay, ignore_index=True)
    d_immed = pd.concat(d_immed, ignore_index=True)
    d_maip = pd.concat(d_maip, ignore_index=True)

    d_delay['cnd'] = 'Delay'
    d_immed['cnd'] = 'Long ITI'
    d_maip['cnd'] = 'Short ITI'

    d = pd.concat([d_delay, d_immed, d_maip], ignore_index=True)

    return d


def load_data_exp_2():

    f_names = [
        os.path.join('../data/bin_data_v2', f)
        for f in os.listdir('../data/bin_data_v2') if f.endswith('.csv')
    ]

    d = pd.concat([pd.read_csv(f) for f in f_names])

    d.rename(
        {
            'sub_num': 'sub',
            'condition': 'cnd',
            'num_problems_solved': 'nps',
            'current_trial_prob': 't_prob',
            'current_trial_exp': 't'
        },
        axis=1,
        inplace=True)

    d.loc[d['cnd'] == 'delay', 'cnd'] = 'Delay'
    d.loc[d['cnd'] == 'immediate', 'cnd'] = 'Long ITI'

    return d


def inspect_duration_vs_trials(d):

    # dd = d[d['nps'] < 14].copy()
    dd = d.copy()

    dd = dd.groupby(['exp', 'cnd', 'sub'])[[
        't2c', 'nps', 'trials_completed', 'experiment_duration'
    ]].mean().reset_index()

    sns.set_palette('rocket', 3)
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 4))
    sns.scatterplot(data=dd[dd['exp'] == 'Exp 1'],
                    x='trials_completed',
                    y='experiment_duration',
                    hue='cnd',
                    ax=ax[0, 0])
    sns.scatterplot(data=dd[dd['exp'] == 'Exp 2'],
                    x='trials_completed',
                    y='experiment_duration',
                    hue='cnd',
                    ax=ax[0, 1])
    plt.savefig('../figures/fig_duration_vs_trials.png')
    plt.close()

    sns.set_palette('rocket', 3)
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(10, 10))
    for i, e in enumerate(dd['exp'].unique()):
        de = dd[dd['exp'] == e]
        for j, m in enumerate(['t2c', 'nps']):
            sns.scatterplot(data=de,
                            x='trials_completed',
                            y=m,
                            hue='cnd',
                            ax=ax[i, j])
            ax[i, j].set_title(e)
    plt.savefig('../figures/fig_trials_vs_meas.png')
    plt.close()


def report_gmm(X, fig_title, fig_name):

    gmm_1 = GaussianMixture(n_components=1, random_state=1).fit(X)
    gmm_2 = GaussianMixture(n_components=2, random_state=1).fit(X)

    print("\n")
    print(f"AIC for 1 component: {gmm_1.aic(X)}\n")
    print(f"AIC for 2 components: {gmm_2.aic(X)}\n")
    print("\n")
    print(f"BIC for 1 component: {gmm_1.bic(X)}\n")
    print(f"BIC for 2 components: {gmm_2.bic(X)}\n")

    # test significance of gmm with LRT
    log_likelihood_1 = gmm_1.score(X) * len(X)
    log_likelihood_2 = gmm_2.score(X) * len(X)
    lrt_statistic = 2 * (log_likelihood_2 - log_likelihood_1)
    degrees_of_freedom = 3 * (2 - 1)
    p_value = chi2.sf(lrt_statistic, degrees_of_freedom)

    print(f"LRT statistic: {lrt_statistic}")
    print(f"Degrees of freedom: {degrees_of_freedom}")
    print(f"P-value: {p_value}")

    # visualise gmm fit
    x = np.linspace(min(X), max(X), 1000)
    logprob_1 = gmm_1.score_samples(x.reshape(-1, 1))
    logprob_2 = gmm_2.score_samples(x.reshape(-1, 1))
    pdf_1 = np.exp(logprob_1)
    pdf_2 = np.exp(logprob_2)

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fs_legend = 14
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(10, 5))
    ax[0, 0].hist(X, bins=50, density=True, alpha=0.6)
    ax[0, 0].plot(x, pdf_1, '-k', label='1 component')
    ax[0, 0].plot(x, pdf_2, '-r', label='2 components')
    ax[0, 0].set_title(fig_title, fontsize=fs_title)
    ax[0, 0].set_xlabel('Trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_ylabel('Proportion of participants', fontsize=fs_axis_label)
    ax[0, 0].legend(fontsize=fs_legend)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    plt.savefig('../figures/' + fig_name)
    plt.close()

    return gmm_1, gmm_2


def report_exp_1(dd):

    X = dd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_1_gmm.png')

    pred = gmm_2.predict(X)
    dd['pred'] = pred

    ddd = dd.loc[dd['pred'] == 0].copy()

    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.375)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks())
    ax[0, 0].set_xticklabels(['Delayed FB', 'Long ITI', 'Control'],
                             fontsize=fs_axis_ticks)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].set_xticks(ax[0, 1].get_xticks())
    ax[0, 1].set_xticklabels(['Delayed FB', 'Long ITI', 'Control'],
                             fontsize=fs_axis_ticks)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_1_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])


def report_exp_2(dd):

    X = dd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_2_gmm.png')

    pred = gmm_2.predict(X)
    dd['pred'] = pred

    # GMM doesn't support 2 components
    # ddd = dd.loc[dd['pred'] == 0].copy()
    ddd = dd.copy()

    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.25)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_xticks([0, 1])
    ax[0, 0].set_xticklabels(['Delayed FB', 'Long ITI'],
                             fontsize=fs_axis_ticks)
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_xticks([0, 1])
    ax[0, 1].set_xticklabels(['Delayed FB', 'Long ITI'],
                             fontsize=fs_axis_ticks)
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_2_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])


def report_exp_3(dd):

    X = dd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_1_gmm.png')

    pred = gmm_2.predict(X)
    dd['pred'] = pred

    ddd = dd.loc[dd['pred'] == 0].copy()

    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(10, 4))
    plt.subplots_adjust(wspace=0.375)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks())
    ax[0,
       0].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                          fontsize=fs_axis_ticks)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].set_xticks(ax[0, 1].get_xticks())
    ax[0,
       1].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                          fontsize=fs_axis_ticks)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_3_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])


def report_exp_3(dd):

    ddd = dd[dd['cnd'].isin(['Delay', 'Long ITI', 'Short ITI'])].copy()
    X = ddd['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_3_gmm.png')

    pred = gmm_2.predict(X)
    ddd['pred'] = pred

    ddd = ddd.loc[ddd['pred'] == 0].copy()


    # Dual-task condition
    ddd_dual = dd[dd['cnd'].isin(['Dual'])].copy()
    X = ddd_dual['t2c'].to_numpy()
    X = np.reshape(X, (X.shape[0], 1))

    gmm_1, gmm_2 = report_gmm(X, '', 'fig_exp_3_gmm.png')

    pred = gmm_2.predict(X)
    ddd_dual['pred'] = pred

    ddd_dual = ddd_dual.loc[ddd_dual['pred'] == 0].copy()

    # recombine 
    ddd = pd.concat([ddd, ddd_dual], ignore_index=True)

    print(ddd.groupby(['exp', 'cnd'])['sub'].nunique())

    dark_gray_palette = sns.dark_palette("gray", reverse=True, as_cmap=False)
    sns.set_palette(dark_gray_palette)

    fs_title = 18
    fs_axis_label = 16
    fs_axis_ticks = 14
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.375)
    sns.barplot(data=ddd, x='cnd', y='t2c', ax=ax[0, 0])
    ax[0, 0].set_title('', fontsize=fs_title)
    ax[0, 0].set_ylabel('Mean trials to criterion', fontsize=fs_axis_label)
    ax[0, 0].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks())
    ax[0, 0].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                             fontsize=fs_axis_ticks)
    ax[0, 0].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    sns.barplot(data=ddd, x='cnd', y='nps', ax=ax[0, 1])
    ax[0, 1].set_title('', fontsize=fs_title)
    ax[0, 1].set_ylabel('Number of problems solved', fontsize=fs_axis_label)
    ax[0, 1].set_xlabel('', fontsize=fs_axis_label)
    ax[0, 1].set_xticks(ax[0, 1].get_xticks())
    ax[0, 1].set_xticklabels(['Delayed FB', 'Long ITI', 'Control', 'Dual-task'],
                             fontsize=fs_axis_ticks)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=fs_axis_ticks)
    for idx, ax in enumerate(ax.flatten()):
        ax.text(-0.1,
                1.05,
                string.ascii_uppercase[idx],
                transform=ax.transAxes,
                size=20)
    plt.savefig('../figures/fig_exp_3_t2c.png')
    plt.close()

    res = pg.anova(data=ddd, dv='t2c', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='t2c', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])

    res = pg.anova(data=ddd, dv='nps', between='cnd', ss_type=3, effsize='np2')
    print()
    print(res)

    res = pg.pairwise_tests(data=ddd, dv='nps', between='cnd', parametric=True)
    print()
    print(res[['A', 'B', 'T', 'dof', 'p-unc']])
