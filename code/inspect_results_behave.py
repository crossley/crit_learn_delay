"""
What this script does
---------------------
- Loads raw trial-level data for each experiment/condition.
- Derives per-participant summary measures:
    * trials_completed: final trial index reached by each participant
    * experiment_duration: total summed RT (see note below for conversion)
    * prob_num: inferred problem number within a session
    * t2c: trials-to-criterion (trials completed within each problem)
    * nps: number of problems solved (max prob_num)
- Fits mixture models to the distribution of trials-to-criterion (t2c):
    * Truncated Ex-Gaussian mixture models (preferred)
    * (Optionally) Gaussian mixture models (older/alternative)
- Produces figures in ../figures/ and prints stats to stdout.

Notes / assumptions
-------------------
- Problem boundaries are inferred by detecting when t_prob decreases from one
  trial to the next (i.e., new problem begins when t_prob < previous t_prob).
- experiment_duration is computed as sum(rt) per subject and condition,
  then multiplied by 5/60. (This reflects the original design’s timing units.
  If you change the task timing, update this conversion.)
- ExGaussian mixture fitting uses hard bounds [a,b] to match the task’s support
  for trials-to-criterion (see each report_* function for bounds).
"""
from util_func import *
from imports import *

if __name__ == "__main__":
    # Run experiment-level reports. Each report function:
    # 1) loads and cleans the relevant experiment data,
    # 2) aggregates to participant-level outcomes,
    # 3) fits mixture models to trials-to-criterion,
    # 4) optionally excludes the "high t2c" component as outliers,
    # 5) prints inferential stats and saves plots to ../figures/.
    report_exp_1()
    report_exp_2()
    # report_exp_3()  # Enable if Exp 3 data are present locally
