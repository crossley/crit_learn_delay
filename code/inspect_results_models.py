from experiment_imports import *
import model_working_memory
import model_procedural_supervised
import model_procedural_reinforcement

# model_working_memory.fit('../data/delay/', *(3.0, 0.5))
# model_working_memory.fit_validate_ppt()

# model_working_memory.fit()
# model_procedural_supervised.fit()
# model_procedural_reinforcement.fit()

# model_working_memory.psp()
# model_procedural_supervised.psp()
# model_procedural_reinforcement.psp()

model_working_memory.plot_psp()
model_procedural_supervised.plot_psp()
model_procedural_reinforcement.plot_psp()

# model_working_memory.fit_validate()
# model_procedural_supervised.fit_validate()
# model_procedural_reinforcement.fit_validate()
