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
# static_plotter.plot_cost_histories([cost_1,cost_2],start=0,points=True,labels=[r'Abdullah',r'Khamis'])

# directions = vector_generator(2, 4)["directions"]

# d_sqrd = directions * directions

# sum_sq = np.sum(d_sqrd, axis=1)

# norms = np.sqrt(sum_sq)

# directions_normalized = directions / norms[:, np.newaxis]


def vector_generator(num_samples: int, d: int):
    return {"directions": np.random.randn(num_samples, d)}

def normalize_rows(vectors: np.ndarray):
    sq = vectors * vectors             
    sum_sq = np.sum(sq, axis=1)            
    norms = np.sqrt(sum_sq)   
    return vectors / norms[:, np.newaxis] 

def make_w0(n: int):
    w0 = np.zeros(n)
    w0[0] = 1
    return w0

def fraction_descent_directions(n: int, p: int):
    w0 = make_w0(n)

    grad = 2 * w0

    directions = vector_generator(p, n)["directions"]
    directions_normalized = normalize_rows(directions)

    count = 0
    for d in directions_normalized:
        if np.dot(grad, d) < 0:
            count += 1

    return count / p

def curse_of_dimensionality(max_n: int, P_list):

    fractions = {}
    for p in P_list:
        fractions[str(p)] = []

    for p in P_list:
        for n in range(1, max_n + 1):
            frac = fraction_descent_directions(n, p)
            fractions[str(p)].append(frac)

    return fractions


N = 25
P = [10, 100, 1000, 10000]

fractions = curse_of_dimensionality(N, P)
print(fractions)


x = list(range(1, N + 1))

plt.figure()
plt.plot(x, fractions["10"], label="P=10", marker="o")
plt.plot(x, fractions["100"], label="P=100", marker="o")
plt.plot(x, fractions["1000"], label="P=1000", marker="o")
plt.plot(x, fractions["10000"], label="P=10000", marker="o")

plt.xlabel("Dimension N")
plt.ylabel("Fraction of descent directions")
plt.title("Fraction of descent directions vs dimension")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()