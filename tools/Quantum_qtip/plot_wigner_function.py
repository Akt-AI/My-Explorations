import numpy as np
from qutip.wigner import wigner
import qutip
import matplotlib.pyplot as plt

xvec = np.linspace(-5,5,200)
N = 20

rho_coherent = qutip.coherent_dm(N, np.sqrt(2))
rho_thermal = qutip.thermal_dm(N, 2)
rho_fock = qutip.fock_dm(N, 2)

W_coherent = wigner(rho_coherent, xvec, xvec)

W_thermal = wigner(rho_thermal, xvec, xvec)

W_fock = wigner(rho_fock, xvec, xvec)

# plot the results

fig, axes = plt.subplots(1, 3, figsize=(12,3))

cont0 = axes[0].contourf(xvec, xvec, W_coherent, 100)

lbl0 = axes[0].set_title("Coherent state")

cont1 = axes[1].contourf(xvec, xvec, W_thermal, 100)

lbl1 = axes[1].set_title("Thermal state")

cont0 = axes[2].contourf(xvec, xvec, W_fock, 100)

lbl2 = axes[2].set_title("Fock state")

plt.show()
#plt.save("wigner.jpg")
