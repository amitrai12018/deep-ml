def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving



    """
    dictionary={}

    if len(data)==0:
        return []

    for sample in data: 
        dictionary[sample[1]]=dictionary.get(sample[1],0)+1
    
    min_count=min(list(dictionary.values()))
    # print(min_count)
    result=[]

    iter_dictionary={}

    for sample in data: 
        if iter_dictionary.get(sample[1],0)<min_count: 
            iter_dictionary[sample[1]]=iter_dictionary.get(sample[1],0)+1

            result.append(sample)

    
    return result
        




