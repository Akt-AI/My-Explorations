# Function for taking input of elements of Matrix
def enter(row, col, Matrix):

      #rows

    for i in range(row):
        temp = []
      # list defined to store values

      #columns
        for j in range(col):
           temp.append(int(input(f"Matrix[{i+1}][{j+1}] = ")))

        Matrix.append(temp) #not a part of j loop
'''
Matrix would be in the form A[0][1] .... as follows
'''

# Function to display()

def displayMatrix(rows, cols, Matrix):
    for i in range(rows):
        for j in range(cols):
           # here
            print(Matrix[i][j], end=" ")
        print("\n")

A = []
rowOfOne = int(input("Enter the number of rows of FIRST matrix : \n"))
colOfOne = int(input("Enter the number of columns of FIRST matrix: \n"))

B = []
rowOfTwo = int(input("Enter the number of rows of SECOND matrix : \n"))
colOfTwo = int(input("Enter the number of columns of SECOND matrix : \n"))

# Checking for the equality of the TWO MATRICES
if (rowOfOne != colOfTwo):
    print("Cannot multiply...")
else:
    print("Enter Elements for the FIRST matrix : ")
    #calling the enter(arg,arg1,arg2) function to take user input for FIRST matrix
    enter(rowOfOne, colOfOne, A)

    print("\n----- : FIRST MATRIX BELOW : ----")
    print("\n\n")
   #calling displayMatrix(arg,arg1,arg2) to display the matrix passed
    displayMatrix(rowOfOne, colOfOne, A)

    print("Enter Elements for the SECOND matrix : ")
    # calling the enter(arg,arg1,arg2) function to take user input for SECOND matrix
    enter(rowOfTwo, colOfTwo, B)

    print("\n----- : SECOND MATRIX BELOW : ----")
    print("\n\n")
    #calling displayMatrix(arg,arg1,arg2) to display the matrix passed

    displayMatrix(rowOfTwo, rowOfTwo, B)

# Initializing the Resultant Matrix
Res = [[0 for i in range(colOfTwo)] for j in range(rowOfOne)]

# Multiplying Two matrices
for i in range(rowOfOne):
    for j in range(colOfTwo):
        Res[i][j] = 0
        #defined the Resultant matrix of size 0
        for k in range(rowOfTwo):
         #mulitplying the corresponding elements of the two matrices and storing it in the Resultant Matrix (Res).
            Res[i][j] += A[i][k] * B[k][j];


# Displaying the Resultant Matrix

print("\n----- : SECOND MATRIX BELOW : ----")
print("\n\n")
displayMatrix(rowOfOne, colOfTwo, Res)
'''
@author: Ayush Raj
Check Condition : Equality of MATRICES
Program : To multiply two matrices without using numpy 
'''