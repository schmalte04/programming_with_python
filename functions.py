import pandas as pd
import numpy as np
import math

def ClassifierFunction(ToClassify, IdealFunction):

    for i in range(2,5):
    
        for x_point in range(100):

            test_point_deviation = abs(np.subtract(ToClassify.y[x_point],ToClassify.iloc[x_point,i]))
            

            if test_point_deviation.max()<IdealFunction.iloc[2,i-2]:        
                ToClassify.iloc[x_point,6] = IdealFunction.iloc[0,i-2]
                ToClassify.iloc[x_point,7] = test_point_deviation.max()
            
            
    return(ToClassify)
