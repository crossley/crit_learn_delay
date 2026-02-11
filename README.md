# crit_learn_delay

This repository contains the code, data, and writing for the
paper:

**Criterial Learning and Feedback Delay: Insights from
Computational Models and Behavioral Experiments**

**Authors:**  
Matthew J. Crossley  
Benjamin O. Pelzer  
F. Gregory Ashby  

## Directory Structure

- **data/**  
  Contains the raw data from the experiments.

- **output/**  
  This is the directory where the output from the parameter
  space partitioning analysis is written. These files can be
  too large for GitHub, so they are not included in the
  repository.

- **code/**  
  Contains the code for the computational models and for
  analyzing the behavioral data.

- **figures/** (generated)  
  Figures produced by the behavioral analysis scripts are
  saved here (paths in code use `../figures/`).

## How to Run

- To analyze the behavioral data, run:  
  `inspect_results_behave.py`

- To analyze the computational model results, run:  
  `inspect_results_model.py`

---

## Behavioral analysis overview (what `inspect_results_behave.py` does)

At a high level, the behavioral pipeline:

1. Loads trial-level data for each experiment and condition
   (see “Raw data formats” below).

2. Infers problem boundaries using resets in `t_prob` (a new
   problem starts when `t_prob` decreases relative to the
   previous trial).

3. Computes participant-level outcome variables (e.g.,
   trials-to-criterion and number of problems solved).

4. Fits 1- vs 2-component mixture models to the distribution
   of trials-to-criterion (`t2c`):
   - primary: truncated Ex-Gaussian mixture models (to
     capture skew/heavy tails)
   - optional/legacy: Gaussian mixture models (GMM)

5. Prints model-fit statistics and inferential tests to
   stdout and writes figures to `../figures/`.

### Derived variables (created during loading / preprocessing)

After loading, the code constructs:

- `prob_num`: inferred problem number within each subject ×
  condition  Computed by counting resets in `t_prob`:
  `(t_prob < previous t_prob).cumsum() + 1`

- `t2c`: “trials to criterion” per problem  
  Implemented as the number of trials within each `(cnd,
  sub, prob_num)` group.

- `nps`: number of problems solved  
  Implemented as `max(prob_num)` within each `(cnd, sub)`
  group.

- `trials_completed`: total trials completed in the session  
  Implemented as `max(t)` within each `(cnd, sub)` group.

- `experiment_duration`: duration proxy  
  Implemented as `sum(rt)` within each `(cnd, sub)` group,
  then scaled by `* 5 / 60`.

---

## Raw data formats and column definitions

### Experiments 1 and 3 (TSV files; `data/delay`, `data/immed`, `data/maip`, `data/dual`)

Files are tab-separated with **no header**. For most
conditions, the loader reads 8 columns:

| Column | Name    | Meaning |
|-------:|---------|---------|
| 1 | `t`      | Trial index within the session. |
| 2 | `t_prob` | Trial index within the current problem (resets at the start of a new problem). |
| 3 | `bnd`    | Boundary parameter for the current problem (task-specific; used for design/inspection plots). |
| 4 | `cat`    | Category label for the stimulus/response (task-specific). |
| 5 | `x`      | Stimulus feature 1 (task-specific; e.g., thickness). |
| 6 | `y`      | Stimulus feature 2 (task-specific; e.g., angle). |
| 7 | `rsp`    | Participant response code. |
| 8 | `rt`     | Response time for the trial (units as recorded by the task). |

**Dual-task condition (Exp 3 only)**: `data/dual` TSV files
contain extra columns; the loader reads:

| Column | Name | Meaning |
|-------:|------|---------|
| 1–8 | same as above | |
| 9–14 | `V8`…`V12`, `rt2` | Additional dual-task outputs (currently ignored in the main behavioral analyses). |

After reading, the loader assigns condition labels (`cnd`) as:
- `Delay`
- `Long ITI`
- `Short ITI`
- `Dual`

It also assigns subject IDs (`sub`) by file order within each condition directory (see note below).

### Experiment 2 (CSV files; `data/bin_data_v2/*.csv`)

Experiment 2 data are read from CSV files with headers, then renamed to match the unified schema:

| Original column | Renamed to | Meaning |
|---|---|---|
| `sub_num` | `sub` | Subject identifier. |
| `condition` | `cnd` | Condition string (`delay` or `immediate`, later mapped to `Delay` / `Long ITI`). |
| `num_problems_solved` | `nps` | Number of problems solved. |
| `current_trial_prob` | `t_prob` | Trial index within the current problem. |
| `current_trial_exp` | `t` | Trial index within the session. |

Condition mapping performed by the code:
- `delay` → `Delay`
- `immediate` → `Long ITI`


