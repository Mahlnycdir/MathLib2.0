import numpy as np 
import matplotlib.pyplot as plt

from mathlibpkg.analysis.integral_1D import Integral1D
from mathlibpkg.analysis.ordinary_dgl import ODGL
from mathlibpkg.analysis.function import Function

def func(t, *args):
    return 2*t


if __name__ == "__main__":
    Integral = Integral1D.improper_integral(func, N_interval=1000, interval = [0,1,3,5])
    print(Integral1D.simple(func, 0, 2, N=1000))
    ODGL.RK4(func, N = 500, h=0.001, y_zero=1, t_zero=0)
    plt.plot(Integral[0], Integral[1])
    plt.show()
# 
    I = np.linspace(0,2*np.pi, 1000)
    func = Function(list(I), list(np.sin(I)))
    print("1", func(np.pi/2))
    print("2", func(np.pi))
    
    
    print("Monte: ", Integral1D.monte_carlo_integral(func, N_samples= 1000, interval_x=[0,2], interval_y=[0,5]))