from experiment_imports import *


def simulate(x, *args):
    alpha = x[0]
    eta_stim = x[1]
    eta_crit = x[2]

    delay = args[0]
    iti = args[1]

    num_consecutive_correct = 0
    t2c = 0
    max_trials = 200

    simulated_data = {
        'cat': [],
        'x': [],
        'y': [],
        'rsp': [],
        'pe': [],
        'prob': []
    }

    num_bins = 7
    bin_width = 14
    bin_bounds = np.arange(0, 100, bin_width)
    bins = np.zeros((num_bins, 2))
    for i in range(bin_bounds.shape[0] - 1):
        bins[i, 0] = bin_bounds[i]
        bins[i, 1] = bin_bounds[i + 1]

    bin_dim = np.arange(0, num_bins, 1)
    np.random.shuffle(bin_dim)

    for prob in range(num_bins):
        x_range = bins[bin_dim[prob]]
        xc_true = np.random.uniform(x_range[0], x_range[1])
        lb1 = x_range[0]
        ub1 = xc_true - 0.1 * bin_width
        lb2 = xc_true + 0.1 * bin_width
        ub2 = x_range[1]
        xc = np.random.uniform(x_range[0], x_range[1])

        for i in range(max_trials):

            # sample stimulus for current trial
            x1 = np.random.uniform(lb1, ub1)
            x2 = np.random.uniform(lb2, ub2)
            x = np.random.choice([x1, x2])

            # memory for the criterion drifts during the ITI
            crit_noise = np.random.normal(0, 1)
            xc += eta_crit * iti * crit_noise

            if x < xc_true:
                cat = 1
            else:
                cat = 2

            if x - xc < 0:
                rsp = 1
            else:
                rsp = 2

            # the criterion and the stimulus drift over the feedback delay
            stim_drift = np.random.normal(0, 1)
            crit_drift = np.random.normal(0, 1)
            x += eta_stim * delay * stim_drift
            xc += eta_crit * delay * crit_drift

            if cat != rsp:
                xc += alpha * (x - xc)
                num_consecutive_correct = 0
            else:
                num_consecutive_correct += 1

            t2c += 1

            simulated_data['cat'].append(cat)
            simulated_data['x'].append(x)
            simulated_data['y'].append(0)
            simulated_data['rsp'].append(rsp)
            simulated_data['pe'].append(alpha * (x - xc))
            simulated_data['prob'].append(prob)

            if num_consecutive_correct >= 12:
                break

    return ({
        'simulated_data': simulated_data,
        't2c': t2c,
        'x_range': x_range,
        'xc_true': xc_true
    })


def fit_obj_func(x, *args):

    alpha = x[0]
    eta_stim = x[1]
    eta_crit = x[2]

    data = args[0]
    delay = args[1]
    iti = args[2]

    num_trials = data.shape[0]
    stim_range_x = data['x'].max() - data['x'].min()
    stim_range_y = data['y'].max() - data['y'].min()
    stim_range = np.max([stim_range_x, stim_range_y])

    xc = np.random.uniform(0, stim_range, 1)
    z = np.zeros(num_trials)

    for trial in range(num_trials):

        # memory for the criterion drifts during the ITI
        crit_noise = np.random.normal(0, 1)
        xc += eta_crit * iti * crit_noise

        # get current stimulus
        if stim_range_x > stim_range_y:
            x = data['x'][trial]
        else:
            x = data['y'][trial]

        # memory for both the criterion and the stimulus drift over the feedback delay
        stim_drift = np.random.normal(0, 1)
        crit_drift = np.random.normal(0, 1)
        x += eta_stim * delay * stim_drift
        xc += eta_crit * delay * crit_drift

        # compute the liklihood of observing the human rsponse given the current model
        sd_stim = eta_stim * delay
        sd_crit = np.sqrt((eta_crit * iti)**2 + (eta_crit * delay)**2)
        sd_total = np.sqrt(sd_stim**2 + sd_crit**2)
        z[trial] = (x - xc) / sd_total

        if data['cat'][trial] != data['rsp'][trial]:
            xc += alpha * (x - xc)

    # z_limit is the z-score value beyond which one should truncate
    z_limit = 7.0
    z[z > z_limit] = z_limit
    z[z < -z_limit] = -z_limit

    # compute negative log likelihood
    A_ind = np.where(np.array(data['rsp']) == 1)
    B_ind = np.where(np.array(data['rsp']) == 2)

    prob_A = sps.norm.cdf(z[A_ind], 0, 1)
    prob_B = 1 - sps.norm.cdf(z[B_ind], 0, 1)

    log_prob_A = np.log(prob_A)
    log_prob_B = np.log(prob_B)

    nll = -(np.sum(log_prob_A) + np.sum(log_prob_B))

    return (nll)


def fit_validate_grid():

    #Generate values for each parameter
    alpha = np.arange(0.01, 1, .5)
    eta_stim = np.arange(0.01, 5, 1)
    eta_crit = np.arange(0.01, 5, 1)

    # all possible combinations of parameters.
    p = [alpha, eta_stim, eta_crit]
    x = np.array(list(product(*p)))

    delay = 3.0
    iti = 0.5
    data = simulate(x, args)
    data = data[1]
    args = (data, delay, iti)

    for i in range(x.shape[0]):
        print(x[i, :])
        res = fit_validate_func(x[i, :], args)
        with open('../output/fit_validate_' + str(i) + '_.csv',
                  'wb') as outfile:
            np.savetxt(res)

    # #Generate processes equal to the number of cores
    # p = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=p)

    # #Distribute the parameter sets evenly across the cores
    # res = pool.map(fit_validate_func, params_in)


def fit_validate_func(x, *args):

    bounds = ((0.0, 1), (0.0, 5), (0.0, 5))
    result = differential_evolution(func=fit_obj_func,
                                    bounds=bounds,
                                    args=args,
                                    disp=False,
                                    polish=False,
                                    maxiter=300,
                                    updating='deferred',
                                    workers=-1)

    res_dict = {
        'alpha': [x[0]],
        'eta_stim': [x[1]],
        'eta_crit': [x[2]],
        'alpha_fit': [res[0][0]],
        'eta_stim_fit': [res[0][1]],
        'eta_crit_fit': [res[0][2]]
    }

    return (res_dict)


def fit_validate_ppt():
    dir_fit = '../output/'
    files = os.listdir(dir_fit)

    # TODO: would it be better if the fit routine output a single file?

    # TODO: change this depending on file type
    files = [x for x in files if 'fit_delay' in x]

    # TODO: change this depending on the file
    delay = 3.0
    iti = 0.5
    args = (delay, iti)

    for ii in range(len(files)):
        f = files[ii]
        fit = np.loadtxt(dir_fit + f)
        x = fit[0:3]

        data = simulate(x, *args)
        data = pd.DataFrame(data['simulated_data'])

        n_probs = data['prob'].unique().shape[0]
        res_record = np.zeros((n_probs, x.shape[0]))

        for prob in range(n_probs):

            args_2 = (data, delay, iti)
            bounds = ((0, 1), (0, 100), (0, 100))
            start_time = time.time()
            res = differential_evolution(func=fit_obj_func,
                                         bounds=bounds,
                                         args=args_2,
                                         disp=False,
                                         polish=False,
                                         maxiter=200,
                                         updating='deferred',
                                         workers=-1)
            end_time = time.time()
            print(end_time - start_time)

            res_record[prob, :] = res['x']

        x = x.reshape(1, x.shape[0])
        res = np.vstack((x, res_record))
        with open('../output/fit_validate_ppt_' + str(ii) + '_.csv',
                  'wb') as outfile:
            np.savetxt(outfile, res)


def fit(dir_data, *args):
    files = os.listdir(dir_data)
    files = files[0:2]

    delay = args[0]
    iti = args[1]

    names = ['t', 't_prob', 'bnd', 'cat', 'x', 'y', 'rsp', 'rt']
    for f in files:
        if '.txt' in f:
            sub = f[0:-4]
            x_obs = np.loadtxt(dir_data + f)
            x_obs = pd.DataFrame(x_obs, columns=names)

            prob = np.zeros(x_obs.shape[0])
            for i in range(x_obs.shape[0]):
                if x_obs['t_prob'][i] == 1.0:
                    prob[i] = prob[i - 1]
                    prob[i] += 1
                else:
                    prob[i] = prob[i - 1]

            x_obs['prob'] = prob

            for prob in x_obs['prob'].unique():

                data = x_obs[x_obs['prob'] == prob]
                data.reset_index(inplace=True)

                args = (data, delay, iti)
                bounds = ((0, 1), (0, 5), (0, 5))
                start_time = time.time()
                result = differential_evolution(func=fit_obj_func,
                                                bounds=bounds,
                                                args=args,
                                                disp=False,
                                                polish=False,
                                                maxiter=200,
                                                updating='deferred',
                                                workers=-1)
                end_time = time.time()
                print(end_time - start_time)

                result = np.append(result['x'], result['fun'])
                with open(
                        '../output/fit_' + sub + '_prob_' + str(int(prob)) +
                        '.csv', 'wb') as outfile:
                    np.savetxt(outfile, result)


def psp():

    #Generate values for each parameter
    alpha = np.arange(0.01, 1, .1)
    eta_stim = np.arange(0.01, 5, .05)
    eta_crit = np.arange(0.01, 5, .05)
    problem = np.arange(1, 100, 1)
    max_trials = 200
    stim_range = 25

    #Generate a list of tuples where each tuple is a combination of parameters.
    #The list will contain all possible combinations of parameters.
    paramlist = list(itertools.product(alpha, eta_stim, eta_crit, problem))
    paramlist = [x + (max_trials, stim_range) for x in paramlist]

    #Generate processes equal to the number of cores
    p = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=p)

    #Distribute the parameter sets evenly across the cores
    res = pool.map(psp_func, paramlist)

    # write results to csv
    psp_write(res)

    # too large to store on github so do some pre processing here
    param_record = pd.read_csv('../output/param_record_working_memory.csv')

    param_record = param_record.groupby(
        ['eta_stim', 'eta_crit',
         'alpha'])[['t2c_delay', 't2c_liti', 't2c_siti']].mean().reset_index()

    param_record.to_csv('../output/param_record_working_memory.csv', index=False)


def psp_func(params):

    alpha = params[0]
    eta_stim = params[1]
    eta_crit = params[2]
    problem = params[3]
    max_trials = params[4]
    stim_range = params[5]

    print(alpha, eta_stim, eta_crit, problem)

    stim = np.random.uniform(0, stim_range, max_trials)
    xc_init = np.random.uniform(0, stim_range, 1)
    xc_true = stim_range / 2.0

    t2c_delay = simulate(xc_init, xc_true, alpha, eta_stim, eta_crit, 3.0, 0.5,
                         stim)
    t2c_liti = simulate(xc_init, xc_true, alpha, eta_stim, eta_crit, 0.5, 3.0,
                        stim)
    t2c_siti = simulate(xc_init, xc_true, alpha, eta_stim, eta_crit, 0.5, 0.5,
                        stim)

    res_dict = {
        'alpha': [alpha],
        'eta_stim': [eta_stim],
        'eta_crit': [eta_crit],
        'problem': [problem],
        't2c_delay': [np.mean(t2c_delay)],
        't2c_liti': [np.mean(t2c_liti)],
        't2c_siti': [np.mean(t2c_siti)]
    }

    return res_dict


def psp_write(res):
    print(res)
    row = 0
    with open('../output/param_record_working_memory.csv', 'wb') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(res[0].keys())
        for i in res:
            writer.writerows(zip(*i.values()))
            row += 1
            if row % 100 == 0:
                print(i)


def plot_psp():

    param_record = pd.read_csv('../output/param_record_working_memory.csv')

    # param_record = param_record.groupby(
    #     ['eta_stim', 'eta_crit',
    #      'alpha'])[['t2c_delay', 't2c_liti', 't2c_siti']].mean().reset_index()

    param_record[['t2c_delay', 't2c_liti', 't2c_siti']] = param_record[[
        't2c_delay', 't2c_liti', 't2c_siti'
    ]] / param_record[['t2c_delay', 't2c_liti', 't2c_siti']].max()

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

    # select every 20th row but sampled equally from each
    # group of psp_clas
    param_record = param_record.groupby('psp_class').apply(
        lambda x: x.iloc[::20]).reset_index(drop=True)

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
                 value_vars=['alpha', 'eta_stim', 'eta_crit'],
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
    ax.set_xticklabels([r'$\eta_{stim}$', r'$\eta_{crit}$', r'$\alpha$'])
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

        ax.scatter(pr['eta_stim'],
                   pr['eta_crit'],
                   pr['alpha'],
                   color=color,
                   marker='o',
                   label=f'Class {cls}',
                   alpha=.5)

    ax.set_xlabel(r'$\eta_{stim}$', fontsize=16)
    ax.set_ylabel(r'$\eta_{crit}$', fontsize=16)
    ax.set_zlabel(r'$\alpha$', fontsize=16)
    ax.set_xlim(param_record['eta_stim'].min(), param_record['eta_stim'].max())
    ax.set_ylim(param_record['eta_crit'].min(), param_record['eta_crit'].max())
    ax.set_zlim(param_record['alpha'].min(), param_record['alpha'].max())
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_zticks(np.arange(0, 1.1, 0.1))
    ax.view_init(elev=15, azim=45)

    plt.savefig('../figures/model_working_memory.pdf')
    plt.close()


def simulate_predicted_bold():

    # All units here in seconds
    peak_delay = 8.0
    peak_disp = 2.0
    under_delay = 16.0
    under_disp = 2.0
    p_u_ratio = 2.0
    hrf_duration = 20

    sample_rate = 1000  # samples per second
    t = np.arange(0, 20, 1 / sample_rate)

    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t, peak_delay / peak_disp, loc=0, scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio

    plt.plot(t, hrf)
    plt.savefig('../figures/hrf.pdf')
    plt.close()

    mp = [0.1, 0.01, 0.01]
    ep = [40, 200, 3.0, 1.0]
    res = model_working_memory.simulate(mp, ep)

    pe = np.array([x[0] for x in res['simulated_data']['pe']])
    n_ev = pe.shape[0]
    trial_duration = 4000  # ms
    ev_duration = 1000  # ms
    sample_rate = 1  # samples per ms
    n_samples_per_trial = trial_duration * sample_rate
    n_sanmple_per_ev = ev_duration * sample_rate
    t_end = n_ev * trial_duration
    t = np.arange(0, t_end, 1 / sample_rate)
    ev = np.zeros(t.shape[0])
    ev_times = np.arange(0, n_ev * trial_duration, trial_duration)

    for i in range(n_ev):
        ev[ev_times[i]:(ev_times[i] + ev_duration)] = pe[i]

    bold = np.convolve(hrf, ev)

    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    ax[0].plot(hrf)
    ax[1].plot(ev)
    ax[2].plot(bold)
    fig.tight_layout()
    plt.savefig('../figures/predicted_bold.pdf')
    plt.close()
