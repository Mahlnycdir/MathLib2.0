import numpy as np 
import matplotlib.pyplot as plt

class ODGL:

    @staticmethod
    def RungeKutta(func_2D, h = 0.01, N = 100, y_zero=0, t_zero=0) -> (list,list):
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
        

class Integral:
    """
    A collection of different integral methods and helpful 
    methods to plot and calculate. 
    """
    @staticmethod
    def simple(func_1D, x_min, x_max, N = 100) -> float:
        """
        Just a simple integral calculator for 1D functions. 
        Nothing special. 
        """
        if x_max < x_min:
            x_min, x_max = x_max, x_min

        x = np.linspace(x_min,x_max,N)
        dx = np.abs(x_max - x_min) / N

        I = 0 
        for x_val in x:
            I += func_1D(x_val)*dx

        return I


    @staticmethod
    def improper_integral(func_1D, N, interval=[0.0, 1.0]) -> (list, list):
        """
        F(t,y) == F(t) = dy/dt -> F(t) is the function integrated over.
        y is the solution array. 
        Remark: F(t) needs to be defined properly with multiple variables,
        but only depending on t. 
        """
        interval = sorted(interval)
        
        if len(interval) > 2:
            # seperate it into multiple intervals and calling this function for every interval
            # afterwards we appending all upcoming y- and t-arrays into one.
            intervals = []
            
            for i in range(1, len(interval)):
                intervals.append([interval[i-1],interval[i]])

            y_list = []
            x_list = []
            step = None
            t_0 = intervals[0][0]
            y_0 = func_1D(t_0, 0)
            
            for inval in intervals:
                if step is not None: 
                    t_0 = step[0][len(step[0]) - 1]
                    y_0 = step[1][len(step[1]) - 1]
                step = ODGL.RungeKutta(func_1D, h=np.abs(inval[1]-inval[0])/N, N=N, y_zero=y_0, t_zero=t_0)
                x_list.append(step[0])
                y_list.append(step[1])

            # The lists are now lists of lists. Hence,
            # we connect them to one array to avoid artifacts. 
            x_list = np.concatenate(x_list)
            y_list = np.concatenate(y_list)
                
            return x_list, y_list   

        return ODGL.RungeKutta(func_1D, h=np.abs(interval[1]-interval[0])/N, N=N, y_zero=func_1D(interval[0], 0), t_zero=interval[0])