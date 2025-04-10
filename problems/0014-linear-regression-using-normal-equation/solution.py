import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X=np.array(X)
    y=np.array(y)
    """
    (XTX)-1Xy=a

    X*theta=y
    XTXtheta=XTy
    theta=(XTX)-1XTy

    Xa=y
    a=
    """
    inter=np.matmul(X.T,X)
    inter=np.linalg.inv(inter)
    theta=np.matmul(np.matmul(inter,X.T),y)

	return theta