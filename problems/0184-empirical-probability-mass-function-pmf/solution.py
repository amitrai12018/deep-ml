def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    import numpy as np
    value,count=np.unique(samples,return_counts=True)
    
    freq=[]
    s=0
    for i in range(len(value)): 
        freq.append([value[i],count[i]])
        s+=count[i]

    freq.sort(key=lambda x: x[0])
    pmf=[]

    for i in freq: 
        pmf.append((i[0],i[1]/s))

    return pmf 


