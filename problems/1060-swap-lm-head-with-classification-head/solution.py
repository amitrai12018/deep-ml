import numpy as np

def swap_lm_head_with_classifier(hidden_states, lm_head_weight, lm_head_bias, num_classes, seed=123):
    """
    Replace an LM head with a classification head and compute new logits.
    """
    d={}
    np.random.seed(seed)

    output_shape=[hidden_states.shape[0],hidden_states.shape[1],num_classes]
    new_head_params=num_classes*hidden_states.shape[2]+num_classes
    old_head_params=hidden_states.shape[2]*lm_head_weight.shape[0]+lm_head_weight.shape[0]
    params_delta=new_head_params-old_head_params
    logits=np.matmul(hidden_states,np.random.randn(num_classes,hidden_states.shape[2]).T)*0.02
    bias = np.zeros(num_classes)
    logits+=bias
    return {'output_shape': output_shape, 'new_head_params': new_head_params, 'old_head_params': old_head_params, 'param_delta': params_delta, 'logits': np.round(logits,6).tolist()}
