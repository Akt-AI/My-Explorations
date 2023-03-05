'''
Function to multiply matrix
'''
def matrix_mul(A, B):
    # if rows of A is not equal to cols of B return -1
    if len(A[0]) != len(B):
        return -1
    # initialize result matrix
    result = [[0 for i in range(len(A))] for _ in range(len(B[0]))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

'''
Matrix display function
'''
def mat_disp(m):
    for _, i in enumerate(m):
        print(i)

'''
Main
'''
if __name__ == "__main__":
    m1 = [[1, 5, 3], [3, 4, 45], [5, 66, 77]]
    m2 = [[66, 63, 5], [34, 23, 45], [4, 5, 6]]
    m3 = matrix_mul(m1, m2)
    mat_disp(m3)
