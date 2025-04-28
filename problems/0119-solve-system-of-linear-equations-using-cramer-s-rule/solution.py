import numpy as np

def cramers_rule(A, b):
    ans=0
    det=np.linalg.det(A)
    if det==0:
        return -1
    ans=[]
    A_cpy=[A[i].copy() for i in range(len(A))]
    for i in range(len(A[0])):
        for j in range(len(A)):
            A_cpy[j][i]=b[j]
        # print(A_cpy)
        det1=np.linalg.det(A_cpy)
        ans.append(det1/det)
        A_cpy=[A[i].copy() for i in range(len(A))]
    
    return ans