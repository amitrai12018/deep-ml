import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
    """
    W= randomly

    y_hat=Wx+b

    loss=avg summation over all X (Y-Y_)^2


    del loss/del W=summation over all X 1/2(Y-Y_)*(-X)
    del loss del b=1/2(Y-Y_)
    W=W-alpha*del(loss)/del(W)
    b=b-alpha*del(loss)/del(b)



    """
	m, n = X.shape
	theta = np.zeros((n, 1))
    W=theta
    # Y_=np.matmul(X,W)
    # print(y.shape)
    # Y_=Y_.reshape((3,))
    # print(Y_.shape)
    # Y_=np.matmul(X,W)
    y=y.reshape((3,1))
    # Y_=Y_.reshape((,))
    
    # loss=(y-Y_)**2
    # for wi in w:
        # v=0
        # for ex in X: 
        #     y_
        #     L=
        #     v=L*X[ex][wi]
                
        # v=
        # wi=wi- alpha/m*()
    # print(((y-Y_)*X).shape)
    # del_w=-1/2/m*(np.sum((y-Y_)*X,axis=0))
    # print(del_w.shape,"shape")
    # W=W-alpha*del_w

    # print(X.shape)
    for i i