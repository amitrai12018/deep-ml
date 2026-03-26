def compressed_col_sparse_matrix(dense_matrix):
	"""
	Convert a dense matrix into its Compressed Column Sparse (CSC) representation.

	:param dense_matrix: List of lists representing the dense matrix
	:return: Tuple of (values, row indices, column pointer)
	"""
	vals=[]
	row_idx=[]
	col_ptr=[0]
	count=0

	for i in range(len(dense_matrix[0])): 
		for j in range(len(dense_matrix)): 
			if dense_matrix[j][i]!=0: 
				vals.append(dense_matrix[j][i])
				row_idx.append(j)
				count+=1
		
		col_ptr.append(count)

	return vals, row_idx, col_ptr
			
