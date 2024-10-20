import numpy as np
import scipy as sp
import pandas as pd
from ggplot import *

d = pd.read_csv('fit_validate.csv', header=0)

ggplot(d, aes(x=problem, y=alpha_fit)) +\
    geom_point()
