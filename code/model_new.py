from experiment_imports import *

"""
Model class I: No criterion is learned
- perceptual drift (y/n)
- criterial drift (NA)
- delay-sensitive updating (y/n)

Model class II: Criterion is learned
- perceptual drift (y/n)
- criterial drift (y/n)
- delay-sensitive updating (y/n)
"""


def simulate_class_I(params, args):

    # Unpack params into sigma_perceptual_noise, alpha_actor, and alpha_critic
    sigma_perceptual_noise = params[0]
    alpha_actor = params[1]
    alpha_critic = params[2]
    eta_perceptual_drift = params[3]
    delay_sensitive_update = params[4]

    # Unpack args into xc_true, t_delay, t_iti, and stim
    xc_true = args[0]
    t_delay = args[1]
    t_iti = args[2]
    stim = args[3]

    if delay_sensitive_update:
        t_delay_actor = t_delay
        t_delay_critic = t_delay
    else:
        t_delay_actor = 1
        t_delay_critic = 1

    num_consecutive_correct = 0
    t2c = 0

    # static params
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
            vis_act[i] = vis_amp * math.exp(-(vis_dist_x**2) / sigma_perceptual_noise)

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

        # perceptual drift may occur during the feedback delay
        # TODO: check with Greg for thoughts about this implementation
        if eta_perceptual_drift:
            vis_dist_x = 0.0
            for i in range(0, dim):
                xx = eta_perceptual_drift * t_delay * np.random.normal(0, 1)
                vis_dist_x = xx - i
                vis_act[i] = vis_amp * math.exp(-(vis_dist_x**2) / sigma_perceptual_noise)

            if act_A != 0.0:
                act_A = np.inner(vis_act, w_A)
            else:
                act_B = np.inner(vis_act, w_B)

        # compute outcome
        r = 1.0 if cat == resp else 0.0

        # compute prediction error
        delta = (r - v)

        # update critic
        v += (1 / t_delay_critic) * alpha_critic * delta

        # update actor
        for i in range(0, dim):

            if delta < 0:
                w_A[i] += ( 1 / t_delay_actor) * alpha_actor * vis_act[i] * act_A * delta * w_A[i]
                w_B[i] += ( 1 / t_delay_actor) * alpha_actor * vis_act[i] * act_B * delta * w_B[i]
            else:
                w_A[i] += ( 1 / t_delay_actor) * alpha_actor * vis_act[i] * act_A * delta * (1 - w_A[i])
                w_B[i] += ( 1 / t_delay_actor) * alpha_actor * vis_act[i] * act_B * delta * (1 - w_B[i])

            w_A[i] = np.clip(w_A[i], 0, 1)
            w_B[i] = np.clip(w_B[i], 0, 1)

        if cat != resp:
            num_consecutive_correct = 0
        else:
            num_consecutive_correct += 1

        t2c += 1

        if num_consecutive_correct >= 12:
            break

    return t2c


def simulate_class_II(params, args):

    # Unpack params into sigma_perceptual_noise, alpha_actor, and alpha_critic
    alpha = params[0]
    eta_perceptual_drift = params[1]
    eta_criterion_drift = params[2]
    delay_sensitive_update = params[3]

    # Unpack args into xc_true, t_delay, t_iti, and stim
    xc_true = args[0]
    t_delay = args[1]
    t_iti = args[2]
    stim = args[3]

    if delay_sensitive_update:
        t_delay = t_delay
    else:
        t_delay = 1

    num_consecutive_correct = 0
    t2c = 0

    for x in stim:

        # Determine true category label
        if x < xc_true:
            cat = 1
        else:
            cat = 2

        # Determine response based on current criterion
        if x - xc < 0:
            resp = 1
        else:
            resp = 2

        # the criterion and the stimulus may drift over the feedback delay
        x += eta_perceptual_drift * t_delay * np.random.normal(0, 1)
        xc += eta_criterion_drift * t_delay * crit_drift * np.random.normal(0, 1)

        # The current criterion is updated following errors and the strength 
        # of the update may be modulated by feedback delay
        if cat != resp:
            xc += (1 / t_delay) * (alpha) * (x - xc)
            num_consecutive_correct = 0
        else:
            num_consecutive_correct += 1

        t2c += 1

        # memory for the criterion may drift during the ITI
        xc += eta_criterion_drift * t_iti * np.random.normal(0, 1)

        if num_consecutive_correct >= 12:
            break

    return t2c

def psp_class_I():

    # Simulate 3 conditions: t_delay, Long ITI, Short ITI
    sigma_perceptual_noise = np.arange(1, 10, 1)
    alpha_critic = np.arange(.01, 1, .1)
    alpha_actor = np.arange(.01, .1, .01)
    eta_perceptual_drift = np.arange(0, 1, 0.1)
    delay_sensitive_update = np.array([True, False])

    num_problems = 100
    num_stim_per_problem = 200

    param_record = {
        'sigma_perceptual_noise': [],
        'alpha_actor': [],
        'alpha_critic': [],
        'eta_perceptual_drift': [],
        'delay_sensitive_update': [],
        't2c_delay': [],
        't2c_liti': [],
        't2c_siti': []
    }

    total_iterations = (
        len(sigma_perceptual_noise)
        * len(alpha_actor)
        * len(alpha_critic)
        * len(eta_perceptual_drift)
        * len(delay_sensitive_update)
        * num_problems
        )

    iteration = 1

    for w in sigma_perceptual_noise:
        for a in alpha_actor:
            for c in alpha_critic:
                for e in eta_perceptual_drift:
                    for d in delay_sensitive_update:

                        record_delay = []
                        record_liti = []
                        record_siti = []

                        for problem in range(0, num_problems):

                            bin_width = 14
                            lb1 = 50-bin_width/2
                            ub2 = 50+bin_width/2
                            xc_true = np.random.uniform(lb1, ub2, 1)[0]
                            ub1 = xc_true - 0.1 * bin_width
                            lb2 = xc_true + 0.1 * bin_width

                            stim_lower = np.random.uniform(lb1, ub1, num_stim_per_problem//2)
                            stim_upper = np.random.uniform(lb2, ub2, num_stim_per_problem//2)
                            stim = np.concatenate((stim_lower, stim_upper))
                            stim = np.random.permutation(stim)

                            params = (w, a, c, e, d)
                            args_delay = (xc_true, 3.0, 0.5, stim)
                            args_liti = (xc_true, 0.5, 3.0, stim)
                            args_siti = (xc_true, 0.5, 0.5, stim)

                            t2c_delay = simulate_class_I(params, args_delay)
                            t2c_liti = simulate_class_I (params, args_liti)
                            t2c_siti = simulate_class_I (params, args_siti)

                            record_delay.append(t2c_delay)
                            record_liti.append(t2c_liti)
                            record_siti.append(t2c_siti)

                            max_t2c = max(np.mean(record_delay), np.mean(record_liti), np.mean(record_siti))

                            param_record['sigma_perceptual_noise'].append(w)
                            param_record['alpha_actor'].append(a)
                            param_record['alpha_critic'].append(c)
                            param_record['eta_perceptual_drift'].append(e)
                            param_record['delay_sensitive_update'].append(d)
                            param_record['t2c_delay'].append(np.mean(record_delay) / max_t2c)
                            param_record['t2c_liti'].append(np.mean(record_liti) / max_t2c)
                            param_record['t2c_siti'].append(np.mean(record_siti) / max_t2c)

                            print(100 * iteration / total_iterations, problem, w, c, a, t2c_delay, t2c_liti, t2c_siti)

                            pd.DataFrame(param_record).to_csv("../output/param_record_class_I.csv", index=False)

                            iteration += 1


def psp_class_II():

    # Simulate 3 conditions: t_delay, Long ITI, Short ITI
    alpha = np.arange(.01, .1, .1)
    eta_perceptual_drift = np.arange(0, 1, 0.1)
    eta_criterion_drift = np.arange(0, 1, 0.1)
    delay_sensitive_update = np.array([True, False])

    num_problems = 100
    num_stim_per_problem = 200

    param_record = {
        'alpha': [],
        'eta_perceptual_drift': [],
        'eta_criterion_drift': [],
        'delay_sensitive_update': [],
        't2c_delay': [],
        't2c_liti': [],
        't2c_siti': []
    }

    total_iterations = (
        * len(alpha)
        * len(eta_perceptual_drift)
        * len(eta_criterion_drift)
        * len(delay_sensitive_update)
        * num_problems
        )

    iteration = 1

    for a in alpha:
        for ep in eta_perceptual_drift:
            for ec in eta_criterion_drift:
                for d in delay_sensitive_update:

                    record_delay = []
                    record_liti = []
                    record_siti = []

                    for problem in range(0, num_problems):

                        bin_width = 14
                        lb1 = 50-bin_width/2
                        ub2 = 50+bin_width/2
                        xc_true = np.random.uniform(lb1, ub2, 1)[0]
                        ub1 = xc_true - 0.1 * bin_width
                        lb2 = xc_true + 0.1 * bin_width

                        stim_lower = np.random.uniform(lb1, ub1, num_stim_per_problem//2)
                        stim_upper = np.random.uniform(lb2, ub2, num_stim_per_problem//2)
                        stim = np.concatenate((stim_lower, stim_upper))
                        stim = np.random.permutation(stim)

                        params = (a, ep, ec, d)
                        args_delay = (xc_true, 3.0, 0.5, stim)
                        args_liti = (xc_true, 0.5, 3.0, stim)
                        args_siti = (xc_true, 0.5, 0.5, stim)

                        t2c_delay = simulate_class_I(params, args_delay)
                        t2c_liti = simulate_class_I (params, args_liti)
                        t2c_siti = simulate_class_I (params, args_siti)

                        record_delay.append(t2c_delay)
                        record_liti.append(t2c_liti)
                        record_siti.append(t2c_siti)

                        max_t2c = max(np.mean(record_delay), np.mean(record_liti), np.mean(record_siti))

                        param_record['alpha'].append(a)
                        param_record['eta_perceptual_drift'].append(ep)
                        param_record['eta_criterion_drift'].append(ec)
                        param_record['delay_sensitive_update'].append(d)
                        param_record['t2c_delay'].append(np.mean(record_delay) / max_t2c)
                        param_record['t2c_liti'].append(np.mean(record_liti) / max_t2c)
                        param_record['t2c_siti'].append(np.mean(record_siti) / max_t2c)

                        print(100 * iteration / total_iterations, problem, w, c, a, t2c_delay, t2c_liti, t2c_siti)

                        pd.DataFrame(param_record).to_csv("../output/param_record_class_II.csv", index=False)

                        iteration += 1


if __name__ == "__main__":

    psp_class_I()
    psp_class_II()
