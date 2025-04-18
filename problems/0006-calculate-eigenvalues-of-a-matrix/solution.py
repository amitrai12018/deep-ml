def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	b=-(matrix[0][0]+matrix[1][1])
    c=matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    # print(b,c)
    sol1=(-b+(b**2-4*c)**(1/2))/2
    sol2=(-b-(b**2-4*c)**(1/2))/2

    return [sol1,sol2]
