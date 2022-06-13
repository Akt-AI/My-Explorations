#Function for taking input of elements of Matrix
def inputElements(row,col,Mat):
    for i in range(row): # Loop for Rows of A
        a=[] #Temp List to store Rows
        for j in range(col): # Loop for Columns of A
            a.append(int(input(f"Mat[{i}][{j}] = ")))
        Mat.append(a)
#Function for Printing Matrix Elements
def printMatrix(rows,cols,Mat):
    for i in range(rows):
        for j in range(cols):
            print(Mat[i][j],end =" ")
        print()

#Defining Empty Matrix A and B & taking sizes from users
A =[]
row1 = int(input("Enter Rows Size of Matrix A : "))
col1 = int(input("Enter Columns Size of Matrix A : "))

B =[]
row2 = int(input("\nEnter Rows Size of Matrix B : "))
col2 = int(input("Enter Columns Size of Matrix B : "))

#Checking the Sizes of both Matrices
if(col1 !=row2):
    print("Matrix Multiplication is Not Possible (Sizes Don't match)")
else:
    print("Enter Elements of Matrix A : ")
    inputElements(row1,col1,A)

    print("\n----- Matrix A -----")

    printMatrix(row1,col1,A)

    print("Enter Elements of Matrix B : ")

    # Taking Elements of Matrix B
    inputElements(row2,col2,B)

    print("\n----- Matrix B -----")

    #Printing Matrix B
    printMatrix(row2,col2,B)

# Initializing the Resultant Matrix
C = [[0 for i in range(col2)] for j in range(row1)]

#Multiplying Two matrices
for i in range(row1): #Loop for Rows of Resultant Matrix
    for j in range(col2): #Loop for Colums of Resultant Matrix
        C[i][j] = 0
        for k in range(row2):
            C[i][j] += A[i][k] * B[k][j];

#Printing the Resultant Matrix
print("\nResultant Matrix ----> ")
printMatrix(row1,col2,C)