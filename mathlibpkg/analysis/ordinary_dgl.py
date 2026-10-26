# import numpy as np 

class ODGL:
    """
    This class is a collection of methods, solving ordinary differential euqations. 
    """
    @staticmethod
    def RK4(func_2D, h = 0.01, N = 100, y_zero=0, t_zero=0):
        """
        dy/dt = f(t, y)
        """
        y_list = []
        t_list = []

        y_list.append(y_zero)
        t_list.append(t_zero)

        for i in range(1, N):
            k1 = func_2D(t_list[i-1],y_list[i-1])
            k2 = func_2D(t_list[i-1]+(h/2),y_list[i-1]+(k1*(h/2)))
            k3 = func_2D(t_list[i-1]+(h/2),y_list[i-1]+(k2*(h/2)))
            k4 = func_2D(t_list[i-1]+h,y_list[i-1]+k3*h)
            
            y_list.append(y_list[i-1] + ((h/6) *(k1 + 2*k2 + 2*k3 + k4)))
            t_list.append(t_list[i-1] + h)

        return t_list, y_list