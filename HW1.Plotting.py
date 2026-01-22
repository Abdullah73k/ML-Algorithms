import sys
sys.path.append("./MLRPlotting")
import matplotlib.pyplot as plt
from MLRPlotting import static_plotter
import numpy as np

# import custom plotting tools
static_plotter = static_plotter.Visualizer();

cost_1 = [1,2,7,9]
cost_2 = [9,6,3,1]
# plot the cost function history for two runs
static_plotter.plot_cost_histories([cost_1,cost_2],start=0,points=True,labels=[r'Abdullah',r'Khamis'])


def vector_generator(num_samples: int, d: int):
    return {"directions": np.random.randn(num_samples, d)}

directions = vector_generator(2, 4)["directions"]

d_sqrd = directions * directions

sum_sq = np.sum(d_sqrd, axis=1)

norms = np.sqrt(sum_sq)

directions_normalized = directions / norms[:, np.newaxis]

print(directions_normalized)