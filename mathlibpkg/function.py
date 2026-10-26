import numpy as np 

class Function: 
    """
    In general, the used numerical methods returning mostly arrays.
    By introducing this class, we want to be able to create a proper 
    function f(x) which can be called instead, and reused afterwards.
    This could be the base for further numerical methods. 
    """
    def __init__(self, x_array = None, y_array = None):
        if x_array is None or y_array is None:
            raise ValueError("You need to implement lists!")
        self.x_array = x_array
        self.y_array = y_array

    def __call__(self, x):
        try:     
            return self.y_array[self.x_array.index(x)]
        except ValueError:
            # Find the smallest interval, where x is included.
            # I.e. min([i1,i2], i1 < x < i2), return a value, weighted 
            # in the interval. 
            x_min = None
            y_min = None
            x_max = None
            y_max = None

            for i in range(len(self.x_array)):
                if x >= self.x_array[i]:
                    x_min = self.x_array[i]
                    y_min = self.y_array[i]
                if x <= self.x_array[i]: 
                    x_max = self.x_array[i]
                    y_max = self.y_array[i]
                    break

            slope = (y_max - y_min)/(x_max - x_min)
            
            return (slope * (x - x_max)) + y_max