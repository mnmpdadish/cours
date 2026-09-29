#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 200)       # create a vector of 200 points.
fig, ax = plt.subplots(figsize=(8, 5))   # initiate canva

ax.plot(x, np.sin(x), label="sin(x)", linewidth=2)  # plot sine
ax.plot(x, np.cos(x), label="cos(x)", linewidth=2)  # plot cosine

ax.set_title("Simple plot")
ax.set_xlabel("x")
ax.set_ylabel("Ψ(x)")
ax.grid(True, alpha=0.3)
ax.legend()

plt.show()

