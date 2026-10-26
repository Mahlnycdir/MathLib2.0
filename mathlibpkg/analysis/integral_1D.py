import numpy as np 
# import matplotlib.pyplot as plt

from mathlibpkg.analysis.ordinary_dgl import ODGL

from random import uniform
        

class Integral1D:
    """
    A collection of different integral (1D) methods and helpful 
    methods to plot and calculate. 
    """
    @staticmethod
    def simple(func_1D, x_min, x_max, N = 100):
        """
        Just a simple Integral1D calculator for 1D functions. 
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
    def improper_integral(func_1D, N_interval, interval=[0.0, 1.0]):
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
                step = ODGL.RK4(func_1D, h=np.abs(inval[1]-inval[0])/N_interval, N=N_interval, y_zero=y_0, t_zero=t_0)
                x_list.append(step[0])
                y_list.append(step[1])

            # The lists are now lists of lists. Hence,
            # we connect them to one array to avoid artifacts. 
            x_list = np.concatenate(x_list)
            y_list = np.concatenate(y_list)
                
            return x_list, y_list   

        return ODGL.RK4(func_1D, h=np.abs(interval[1]-interval[0])/N_interval, N=N_interval, y_zero=func_1D(interval[0], 0), t_zero=interval[0])
    
    
    @staticmethod
    def monte_carlo_integral(func1D, N_samples, interval_x, interval_y):
        """
        A point is given by p = (uniform(x_Min, x_Max, N), np.uniform(x_Min, x_Max, N)). Hence we define the points as matrix P[i][j], but we are only interested in
        the diagonal. 

        Args:
            func1D (funcObj): A function depending on one varaibale.
            N_samples (int): Number of samples (points).
            interval_x (list): The interval, limiting the creation of x values onto this interval.
            interval_y (list): The interval, limiting the creation of y values onto this interval.
        """
        false_points = 0
        right_points = 0
        
        for i in range(N_samples):
            point = (uniform(interval_x[0], interval_x[1]), uniform(interval_y[0], interval_y[1]))
            if point[1] > func1D(point[0]):
                false_points += 1
            else:
                right_points += 1
            
        return (interval_x[1] - interval_x[0]) * (interval_y[1] - interval_y[0]) * right_points / N_samples + interval_y[0] * (interval_x[1] - interval_x[0])