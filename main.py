import numpy as np 
import matplotlib.pyplot as plt

from mathlibpkg.analysis_1D import Integral, ODGL
from mathlibpkg.function import Function

def func(t,y):
    return np.sin(t)


if __name__ == "__main__":
    # integral = Integral.improper_integral(func, N = 1000, interval = [0,1,3,5])
    # print(Integral.simple(func, 0, 2, N=1000))
    # ODGL.RungeKutta(func, N = 500, h=0.001, y_zero=1, t_zero=0)
    # plt.plot(integral[0], integral[1])
    # plt.show()

    I = np.linspace(0,2*np.pi, 1000)
    func = Function(list(I), list(np.sin(I)))
    print("1", func(np.pi/2))
    print("2", func(np.pi))