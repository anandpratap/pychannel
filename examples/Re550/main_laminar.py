import sys
sys.path.insert(1, "../../src")
import matplotlib.pyplot as plt
import numpy as np

from laminar import LaminarEquation
from utils import load_data, load_solution_laminar

dirname ="base_solution"
y, u = load_solution_laminar(dirname)
Retau = 550.0
eqn = LaminarEquation(y, u, Retau)
eqn.writedir = "solution"
eqn.solve()
dns, wilcox = load_data()

plt.figure(1)
plt.semilogx(eqn.yp, eqn.up, 'r-', label=r'$k-\omega$')
plt.semilogx(eqn.yp[::5], eqn.uap[::5], 'bo', label=r'Analytic', mfc="white")
plt.xlabel(r"$y^+$")
plt.ylabel(r"$u^+$")
plt.legend(loc=2)
plt.tight_layout()
plt.show()
