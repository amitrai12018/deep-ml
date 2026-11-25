def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    parameters=0
    
    for i in layers : 
        if i['type']=='dense':
            parameters+=i['input_size']*i['output_size']
            if 'bias' not in i or  i['bias']==True: 
                parameters+=i['output_size']
        else: 
            parameters+=i['in_channels']*i['out_channels']*i['kernel_size']**2
            if 'bias'  not in i or i['bias']==True: 
                parameters+=i['out_channels']

    return parameters