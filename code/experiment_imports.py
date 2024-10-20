import csv
import time
import os
import sys
import random
import math
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brute, fmin, differential_evolution
import scipy.stats as sps
import multiprocessing as mp
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib import rcParams
from matplotlib import rc
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns

# mpl.use('pgf')
# plt.rcParams.update({
#     "pgf.texsystem": "lualatex",
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "pgf.preamble": [
#         "\\usepackage{fontspec}",
#         "\\usepackage{amsmath,amsfonts,amssymb}",
#         "\\usepackage{gensymb}",
#         r"\setmainfont{Arial}"
#          ]
# })
