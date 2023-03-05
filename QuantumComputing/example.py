from qutip import *
import numpy as np
import matplotlib.pyplot as plt 

def hamlitonian():
    H = 2 * np.pi * 0.1 * sigmax()
    return H
    
H = hamlitonian()
psi0 = basis(2, 0)
times = np.linspace(0.0, 10.0, 100)
result = mesolve(H, psi0, times, [], [sigmaz(), sigmay()])
fig, ax = plt.subplots()
ax.plot(result.times, result.expect[0]);
ax.plot(result.times, result.expect[1]);
ax.set_xlabel('Time');
ax.set_ylabel('Expectation values');
ax.legend(("Sigma-Z", "Sigma-Y")); 
plt.show()

