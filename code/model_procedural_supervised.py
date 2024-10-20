from experiment_imports import *


def simulate(xc_init, xc_true, alpha, noise, delay, iti, stim):

    xc = xc_init
    num_consecutive_correct = 0
    t2c = 0

    for x in stim:

        # percept is corrupted by perceptual noise
        noise_err = np.random.normal(0, noise, 1)
        x += noise_err

        if x < xc_true:
            cat = 1
        else:
            cat = 2

        if x - xc < 0:
            resp = 1
        else:
            resp = 2

        if cat != resp:
            xc += (1 / delay) * (alpha) * (x - xc)
            num_consecutive_correct = 0
        else:
            num_consecutive_correct += 1

        t2c += 1

        if num_consecutive_correct >= 12:
            break

    return (t2c)


def fit(fit_params, delay, iti, human_data):

    xc_init = fit_params[0]
    alpha = fit_params[1]
    noise = fit_params[2]

    # human_data must be a dict with keys: 'cat', 'x', 'y', 'resp'
    num_trials = len(human_data['x'])
    stim_range_x = max(human_data['x']) - min(human_data['x'])
    stim_range_y = max(human_data['y']) - min(human_data['y'])
    stim_range = max([stim_range_x, stim_range_y])

    xc = xc_init
    L = []

    for trial in range(num_trials):

        # get current stimulus
        if stim_range_x > stim_range_y:
            x = human_data['x'][trial]
        else:
            x = human_data['y'][trial]

        # percept is corrupted by perceptual noise
        noise_err = np.random.normal(0, noise, 1)
        x += noise_err

        # compute the liklihood of observing the human response given the current model
        L.extend(abs(x - xc) / float(stim_range))

        if human_data['cat'][trial] != human_data['resp'][trial]:
            xc += (1 / delay) * (alpha) * (x - xc)

    return (np.prod(L))


def fit_validate():
    # generate fake data from the model using simulate()
    # find whether or not fit() recovers parameters
    # find constraints on when fit() performs well and not
    pass


def fit_humans():
    # load human data
    # iterate through per human per problem
    # fit each problem and log results
    pass


def psp():

    # Simulate 3 conditions: Delay, Long ITI, Short ITI
    alpha = np.arange(0.01, 1, .1)
    noise = np.arange(0.01, 5, .1)

    num_problems = 100
    num_stim_per_problem = 200

    stim_range = 25

    param_record = {
        'alpha': [],
        'noise': [],
        't2c_delay': [],
        't2c_liti': [],
        't2c_siti': []
    }

    for a in alpha:
        print('alpha = ' + str(a))
        for n in noise:
            record_delay = []
            record_liti = []
            record_siti = []
            for problem in range(0, num_problems):

                stim = np.random.uniform(0, stim_range, num_stim_per_problem)
                xc_init = np.random.uniform(0, stim_range, 1)
                xc_true = stim_range / 2.0

                t2c_delay = simulate(xc_init, xc_true, a, n, 3.0, 0.5, stim)
                t2c_liti = simulate(xc_init, xc_true, a, n, 0.5, 3.0, stim)
                t2c_siti = simulate(xc_init, xc_true, a, n, 0.5, 0.5, stim)

                record_delay.append(t2c_delay)
                record_liti.append(t2c_liti)
                record_siti.append(t2c_siti)

            max_t2c = max(np.mean(record_delay), np.mean(record_liti),
                          np.mean(record_siti))

            param_record['alpha'].append(a)
            param_record['noise'].append(n)
            param_record['t2c_delay'].append(np.mean(record_delay) / max_t2c)
            param_record['t2c_liti'].append(np.mean(record_liti) / max_t2c)
            param_record['t2c_siti'].append(np.mean(record_siti) / max_t2c)

    # write results to csv
    with open("../output/param_record_procedural_supervised.csv",
              "wb") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(param_record.keys())
        writer.writerows(zip(*param_record.values()))


def plot_psp():

    param_record = pd.read_csv(
        '../output/param_record_procedural_supervised.csv')

    param_record['psp_class'] = "Other"

    # delay is greater than long iti and short iti
    param_record.loc[(param_record['t2c_delay'] > param_record['t2c_liti']) &
                     (param_record['t2c_delay'] > param_record['t2c_siti']),
                     'psp_class'] = "Delay impaired"

    # long iti is greater than delay and short iti
    param_record.loc[(param_record['t2c_liti'] > param_record['t2c_delay']) &
                     (param_record['t2c_liti'] > param_record['t2c_siti']),
                     'psp_class'] = "Long ITI impaired"

    # short iti is greater than delay and long iti
    param_record.loc[(param_record['t2c_siti'] > param_record['t2c_delay']) &
                     (param_record['t2c_siti'] > param_record['t2c_liti']),
                     'psp_class'] = "Control impaired"

    # make psp_class an ordered category
    param_record['psp_class'] = pd.Categorical(param_record['psp_class'],
                                               ordered=True,
                                               categories=[
                                                   'Delay impaired',
                                                   'Long ITI impaired',
                                                   'Control impaired', 'Other'
                                               ])

    sns.set_palette("rocket", 4)

    fig = plt.figure(figsize=(10, 8))

    xywhA = [0.1, 0.6, 0.375, 0.3]
    xywhC = [0.1, 0.15, 0.375, 0.3]
    xywhB = [0.55, 0.55, 0.4, 0.45]
    xywhD = [0.55, 0.15, 0.375, 0.3]

    ax1 = fig.add_axes(xywhA)
    ax2 = fig.add_axes(xywhC)
    ax3 = fig.add_axes(xywhB, projection='3d')
    ax4 = fig.add_axes(xywhD)

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
                  data=param_record,
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
        tick.get_text().replace('impaired', '\nimpaired')
        for tick in ax.get_xticklabels()
    ]
    ax.set_xticklabels(labels, fontsize=10)

    ax = ax2
    pr = param_record.copy()
    pr = pr.melt(id_vars='psp_class',
                 value_vars=['alpha', 'noise'],
                 var_name='param')
    pr['value'] = pr.groupby(['param'
                              ])['value'].transform(lambda x: x / x.max())
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
    ax.set_xticks(np.arange(0, 2, 1))
    ax.set_xticklabels([r'$\alpha$', r'$\sigma$'])
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)

    unique_classes = param_record['psp_class'].unique()

    ax = ax3
    for i, cls in enumerate(unique_classes):
        pr = param_record[param_record['psp_class'] == cls]
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=15, azim=45)

    ax = ax4
    for i, cls in enumerate(unique_classes):
        pr = param_record[param_record['psp_class'] == cls]
        color = ['C0', 'C1', 'C2', 'C3'][i]

        ax.scatter(pr['alpha'],
                   pr['noise'],
                   color=color,
                   marker='o',
                   label=f'Class {cls}',
                   alpha=.5)

    ax.set_xlabel(r'$\alpha$', fontsize=16)
    ax.set_ylabel(r'$\sigma$', fontsize=16)
    ax.set_xlim(param_record['alpha'].min(),
                param_record['alpha'].max())
    ax.set_ylim(param_record['noise'].min(),
                param_record['noise'].max())
    ax.set_xticks(np.arange(0, 1.05, 0.1))
    ax.set_yticks(np.arange(0, 6, 1))

    plt.savefig('../figures/model_procedural_supervised.pdf')
    plt.close()
