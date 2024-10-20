from experiment_imports import *


def simulate(xc_true, vis_width, alpha_critic, alpha_actor, delay, iti, stim):

    data_record = {'cat': [], 'x': [], 'y': [], 'resp': []}

    num_consecutive_correct = 0
    t2c = 0

    # static params:
    dim = 25
    w_base = 0.5
    w_noise = 0.01
    vis_amp = 1.0

    # buffers
    vis_act = np.zeros(dim)
    w_A = np.zeros(dim)
    w_B = np.zeros(dim)

    # init
    v = 0.5

    # init weights: Note that we assume fresh weights every problem
    w_A = np.random.normal(w_base, w_noise, dim)
    w_B = np.random.normal(w_base, w_noise, dim)

    for x in stim:

        # Determine true category label
        if x < xc_true:
            cat = 1
        else:
            cat = 2

        # compute input activation via radial basis functions
        vis_dist_x = 0.0
        for i in range(0, dim):
            vis_dist_x = x - i
            vis_act[i] = vis_amp * math.exp(-(vis_dist_x**2) / vis_width)

        # Compute A and B unit activations via dot product
        act_A = 0.0
        act_B = 0.0
        act_A = np.inner(vis_act, w_A)
        act_B = np.inner(vis_act, w_B)

        # Compute resp via max
        discrim = act_A - act_B
        resp = 1 if discrim > 0 else 2

        # Implement strong lateral inhibition
        act_A = act_A if resp == 1.0 else 0.0
        act_B = act_B if resp == 2.0 else 0.0

        # compute outcome
        r = 1.0 if cat == resp else 0.0

        # compute prediction error
        delta = (r - v)

        # update critic
        v += (1 / delay) * alpha_critic * delta

        # update actor
        for i in range(0, dim):

            if delta < 0:
                w_A[i] += (
                    1 /
                    delay) * alpha_actor * vis_act[i] * act_A * delta * w_A[i]
                w_B[i] += (
                    1 /
                    delay) * alpha_actor * vis_act[i] * act_B * delta * w_B[i]
            else:
                w_A[i] += (
                    1 / delay) * alpha_actor * vis_act[i] * act_A * delta * (
                        1 - w_A[i])
                w_B[i] += (
                    1 / delay) * alpha_actor * vis_act[i] * act_B * delta * (
                        1 - w_B[i])

            w_A[i] = cap(w_A[i])
            w_B[i] = cap(w_B[i])

        # # diagnostic plot
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 3, 1)
        # ax.plot(vis_act, linewidth=2.0)
        # ax.set_ylim([0, 1])
        # ax = fig.add_subplot(1, 3, 2)
        # ax.plot(w_A, linewidth=2.0)
        # ax.set_ylim([0, 1])
        # ax = fig.add_subplot(1, 3, 3)
        # ax.plot(w_B, linewidth=2.0)
        # ax.set_ylim([0, 1])
        # plt.show()

        data_record['cat'].append(cat)
        data_record['x'].append(x)
        data_record['y'].append(0.0)
        data_record['resp'].append(resp)

        # problem solved?
        if cat != resp:
            num_consecutive_correct = 0
        else:
            num_consecutive_correct += 1

        t2c += 1

        if num_consecutive_correct >= 12:
            break

    return ([t2c, data_record])


def fit(fit_params, delay, iti, human_data):
    vis_width = fit_params[0]
    alpha_critic = fit_params[1]
    alpha_actor = fit_params[2]

    # human_data must be a dict with keys: 'cat', 'x', 'y', 'resp'
    num_trials = len(human_data['x'])
    stim_range_x = max(human_data['x']) - min(human_data['x'])
    stim_range_y = max(human_data['y']) - min(human_data['y'])
    stim_range = max([stim_range_x, stim_range_y])

    # create an empty list to store liklihoods
    L = []

    # static params:
    dim = 25
    w_base = 0.5
    w_noise = 0.01
    vis_amp = 1.0

    # buffers
    vis_act = np.zeros(dim)
    w_A = np.zeros(dim)
    w_B = np.zeros(dim)

    # init
    v = 0.5

    # init weights: Note that we assume fresh weights every problem
    w_A = np.random.normal(w_base, w_noise, dim)
    w_B = np.random.normal(w_base, w_noise, dim)

    for trial in range(num_trials):

        # get current stimulus and transform to [0 dim]
        if stim_range_x > stim_range_y:
            x = float(dim) * human_data['x'][trial] / float(stim_range)
        else:
            x = float(dim) * human_data['y'][trial] / float(stim_range)

        # compute input activation via radial basis functions
        vis_dist_x = 0.0

        for i in range(0, dim):
            vis_dist_x = x - i
            vis_act[i] = vis_amp * math.exp(-(vis_dist_x**2) / vis_width)

        # Compute A and B unit activations via dot product
        act_A = 0.0
        act_B = 0.0
        act_A = np.inner(vis_act, w_A)
        act_B = np.inner(vis_act, w_B)

        # Compute resp via max
        discrim = act_A - act_B
        resp = 1 if discrim > 0 else 2

        # compute the liklihood of observing the human response given the current model
        # L.append(abs(discrim) / max([act_A, act_B]))
        L.append(abs(discrim))

        # Implement strong lateral inhibition
        act_A = act_A if resp == 1.0 else 0.0
        act_B = act_B if resp == 2.0 else 0.0

        # update critic
        r = 1.0 if human_data['cat'][trial] == human_data['resp'][
            trial] else 0.0
        v += (1 / delay) * alpha_critic * (r - v)

        # update actor
        for i in range(0, dim):
            w_A[i] += (1 / delay) * alpha_actor * (r - v) * act_A
            w_B[i] += (1 / delay) * alpha_actor * (r - v) * act_B

    return (-np.log(np.prod(L)))


def fit_validate():

    n = 10

    p_true_record = []
    p_init_record = []
    p_estimated_record = []

    for i in range(n):
        for j in range(n):
            for k in range(n):

                # generate fake data from the model using simulate()
                p_true = [
                    np.random.uniform(0, 10, 1),
                    np.random.uniform(0, 1, 1),
                    np.random.uniform(0, 1, 1)
                ]
                delay = 1.0
                iti = 1.0

                num_problems = 10
                num_stim_per_problem = 200
                stim_range = 25

                stim = np.random.uniform(0, stim_range, num_stim_per_problem)
                xc_true = stim_range / 2.0

                [t2c, data_record] = simulate(xc_true, p_true[0], p_true[1],
                                              p_true[2], delay, iti, stim)

                # find whether or not fit() recovers parameters
                p_init = [
                    np.random.uniform(0, 10, 1),
                    np.random.uniform(0, 1, 1),
                    np.random.uniform(0, 1, 1)
                ]
                bnds = ((1, 10), (0, 1), (0, 1))

                opt = {
                    'maxiter': 1000,
                    'disp': False,
                    'eps': .1,
                    'ftol': 1e-20,
                    'gtol': 1e-20
                }

                best_fit = minimize(fit,
                                    x0=p_init,
                                    args=(delay, iti, data_record),
                                    method='L-BFGS-B',
                                    bounds=bnds,
                                    options=opt)

                p_true_record.append(p_true)
                p_init_record.append(p_init)
                p_estimated_record.append(best_fit['x'])

    # find constraints on when fit() performs well
    fig = plt.figure(figsize=(18, 9))

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    xs = [x[0] for x in p_true_record]
    ys = [x[0] for x in p_init_record]
    zs = [x[0] for x in p_estimated_record]
    ax.scatter(xs, ys, zs, marker='o', alpha=.7)
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    ax.set_xlabel('True')
    ax.set_ylabel('Init')
    ax.set_zlabel('Estimate')
    ax.view_init(elev=15, azim=-135)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    xs = [x[1] for x in p_true_record]
    ys = [x[1] for x in p_init_record]
    zs = [x[1] for x in p_estimated_record]
    ax.scatter(xs, ys, zs, marker='o', alpha=.7)
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.set_xlabel('True')
    ax.set_ylabel('Init')
    ax.set_zlabel('Estimate')
    ax.view_init(elev=15, azim=-135)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    xs = [x[2] for x in p_true_record]
    ys = [x[2] for x in p_init_record]
    zs = [x[2] for x in p_estimated_record]
    ax.scatter(xs, ys, zs, marker='o', alpha=.7)
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.set_xlabel('True')
    ax.set_ylabel('Init')
    ax.set_zlabel('Estimate')
    ax.view_init(elev=15, azim=-135)

    plt.show()


def fit_humans():
    # load human data
    # iterate through per human per problem
    # fit each problem and log results
    pass


def psp():

    # Simulate 3 conditions: Delay, Long ITI, Short ITI
    vis_width = np.arange(1, 10, 1)
    alpha_critic = np.arange(.01, 1, .1)
    alpha_actor = np.arange(.01, .1, .01)

    num_problems = 100
    num_stim_per_problem = 200

    stim_range = 25

    param_record = {
        'vis_width': [],
        'alpha_critic': [],
        'alpha_actor': [],
        't2c_delay': [],
        't2c_liti': [],
        't2c_siti': []
    }

    for w in vis_width:
        for c in alpha_critic:
            for a in alpha_actor:

                record_delay = []
                record_liti = []
                record_siti = []
                for problem in range(0, num_problems):
                    stim = np.random.uniform(0, stim_range,
                                             num_stim_per_problem)
                    xc_true = stim_range / 2.0

                    t2c_delay = simulate(xc_true, w, c, a, 3.0, 0.5, stim)[0]
                    t2c_liti = simulate(xc_true, w, c, a, 0.5, 3.0, stim)[0]
                    t2c_siti = simulate(xc_true, w, c, a, 0.5, 0.5, stim)[0]

                    record_delay.append(t2c_delay)
                    record_liti.append(t2c_liti)
                    record_siti.append(t2c_siti)

                print(w, c, a, t2c_delay, t2c_liti, t2c_siti)

                max_t2c = max(np.mean(record_delay), np.mean(record_liti),
                              np.mean(record_siti))

                param_record['vis_width'].append(w)
                param_record['alpha_critic'].append(c)
                param_record['alpha_actor'].append(a)
                param_record['t2c_delay'].append(
                    np.mean(record_delay) / max_t2c)
                param_record['t2c_liti'].append(np.mean(record_liti) / max_t2c)
                param_record['t2c_siti'].append(np.mean(record_siti) / max_t2c)

    # write results to csv
    with open("param_record_procedural_reinforcement.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(param_record.keys())
        writer.writerows(zip(*param_record.values()))


def plot_psp():

    param_record = pd.read_csv(
        '../output/param_record_procedural_reinforcement.csv')

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
                 value_vars=['vis_width', 'alpha_actor', 'alpha_critic'],
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
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels(
        [r'$\sigma$', r'$\alpha_{actor}$', r'$\alpha_{critic}$'])
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

        ax.scatter(pr['alpha_actor'],
                   pr['alpha_critic'],
                   pr['vis_width'],
                   color=color,
                   marker='o',
                   label=f'Class {cls}',
                   alpha=.5)

    ax.set_xlabel(r'$\alpha_{actor}$', fontsize=16)
    ax.set_ylabel(r'$\alpha_{critic}$', fontsize=16)
    ax.set_zlabel(r'$\sigma$', fontsize=16)
    ax.set_xlim(param_record['alpha_actor'].min(),
                param_record['alpha_actor'].max())
    ax.set_ylim(param_record['alpha_critic'].min(),
                param_record['alpha_critic'].max())
    ax.set_zlim(param_record['vis_width'].min(),
                param_record['vis_width'].max())
    ax.set_xticks(np.arange(0, 0.25, 0.05))
    ax.set_yticks(np.arange(0, 0.25, 0.05))
    ax.set_zticks(np.arange(0, 10, 1))
    ax.view_init(elev=15, azim=45)

    plt.savefig('../figures/model_procedural_reinforcement.pdf')
    plt.close()


# Utility function
def cap(val):
    val = 0.0 if val < 0.0 else val
    val = 1.0 if val > 1.0 else val
    return val
