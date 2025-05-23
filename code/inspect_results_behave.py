from util_func import *
from imports import *

d = load_data()

inspect_duration_vs_trials(d)

dd = d.groupby(['exp', 'cnd', 'sub'])[['t2c', 'nps']].mean().reset_index()
print(dd.groupby(['exp', 'cnd'])['sub'].nunique())

dd1 = dd.loc[dd['exp'] == 'Exp 1'].copy()
dd2 = dd.loc[dd['exp'] == 'Exp 2'].copy()

report_exp_1(dd1)
report_exp_2(dd2)

d3 = load_data_exp_3()
dd3 = d3.groupby(['exp', 'cnd', 'sub'])[['t2c', 'nps']].mean().reset_index()
print(dd3.groupby(['exp', 'cnd'])['sub'].nunique())
report_exp_3(dd3)
