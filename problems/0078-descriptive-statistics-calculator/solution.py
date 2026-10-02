import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    mean=np.mean(data)
    median=np.median(data)
    value,count=np.unique(data,return_counts=True)
    mode=value[np.argmax(count)]
    variance=np.var(data)
    std_dev=np.std(data)
    q=np.quantile(data,[0.25,0.5,0.75])
    iqr=q[2]-q[0]
    return {"mean":mean,"median":median,"mode":mode,"variance":variance,"standard_deviation":std_dev,"25th_percentile":q[0],"50th_percentile":q[1],"75th_percentile":q[2],"interquartile_range":iqr}

