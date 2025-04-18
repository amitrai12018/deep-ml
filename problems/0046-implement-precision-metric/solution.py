import numpy as np
def precision(y_true, y_pred):
	"""
    precision=TP/(TP+FP)

    """
    pos=0
    tp=0
    for i in range(len(y_pred)):
        if y_pred[i]==1:
            pos+=1
            if y_true[i]==1:
                tp+=1
    return tp/pos
	pass
