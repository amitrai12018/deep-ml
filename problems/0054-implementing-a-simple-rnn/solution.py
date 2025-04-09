import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	"""
    h_t=tanh([Wx,Wh][X,h_t-1]+b)

    3,2
    4,2


    """

    
    Wx=np.array(Wx)
    Wh=np.array(Wh)
    b=np.array(b)
    
    final_hidden_state=np.array(initial_hidden_state)
    for i in range(len(input_sequence)):
        final_hidden_state=np.tanh(np.matmul(Wx,np.array(input_sequence[i]))+np.matmul(Wh,final_hidden_state)+np.array(b))
        

	return final_hidden_state