import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix to its Compressed Row Sparse (CSR) representation.

	:param dense_matrix: 2D list representing a dense matrix
	:return: A tuple containing (values array, column indices array, row pointer array)
	"""
	val_arr=[]
	col_idx=[]
	row_ptr=[0]
	count=0
	for i in range(len(dense_matrix)): 
		
		for j in range(len(dense_matrix[0])): 
			if dense_matrix[i][j]!=0: 
				val_arr.append(dense_matrix[i][j])
				col_idx.append(j)
				count+=1
		row_ptr.append(count)

	return val_arr,col_idx,row_ptr

