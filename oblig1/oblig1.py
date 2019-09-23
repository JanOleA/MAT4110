import numpy as np
import matplotlib.pyplot as plt


def make_A(x, y, m):
    """ Sets up and returns the Vandermonde matrix A for the minimization of
    ||Ax - b||^2
    """
    n = len(y)

    # setting up Vandermonde matrix
    A = np.matrix(np.zeros((n, m)))
    A[:,0] = np.full((n,1), 1) # first column is 1 everywhere

    # set up x-values as a column to fill the columns of A
    x_col = x[:]
    x_col.resize((n,1))

    # loop over columns
    for j in range(1,m):
        A[:,j] = x_col**j

    return A


def cholesky(A):
    """ Returns the Cholesky factorization R of the matrix A where
    A = R*R^T

    Requires A to be a symmetric and positive definite matrix
    """
    n = len(A)

    A = np.matrix(A)

    L = np.matrix(np.zeros((n,n)))
    D = np.matrix(np.zeros((n,n)))

    for k in range(n):
        D[k,k] = A[k,k]
        L[:,k] = A[:,k]/D[k,k]

        A = A - D[k,k]*np.dot(L[:,k], L[:,k].transpose())

    D_root = np.power(D, 0.5)

    R = np.dot(L, D_root)
    return R


def ex1(x, y, m):
    """ Solution to exercise 1 of the obligatory exercise """
    A = make_A(x, y, m)
    n = len(y)

    x_coeffs = np.zeros(m) # polynomial coefficients to find

    Q, R = np.linalg.qr(A)

    # right hand side
    RHS = np.dot(Q.transpose(), y)[0]

    # to solve: Q*R*x = y
    # R*x = Q^T*y
    # apply back substitution to R to find x

    for i in range(0, m)[::-1]:
        sum = 0
        for j in range(i, m):
            sum += R[i, j]*x_coeffs[j]
        x_coeffs[i] = (RHS[0,i]-sum)/R[i,i]

    x_model = np.linspace(x[0], x[-1], 1000)
    model = np.zeros(1000)
    for i, c in enumerate(x_coeffs):
        model += c*x_model**i

    return x_model, model

def ex2(x, y, m):
    A = make_A(x, y, m)
    n = len(y)

    x_coeffs = np.zeros(m) # polynomial coefficients to find
    y_temp = np.zeros(m) # need to find this first

    B = np.dot(A.transpose(), A)
    R = cholesky(B)

    # System is: R*R^T*x = A^T*b
    # First solve R*y = A^T*b where y = R^T*x
    # R is lower triangular, so must use forward substitution

    RHS = np.dot(A.transpose(), y)

    for i in range(0, m):
        sum = 0
        for j in range(0, i):
            sum += R[i, j]*y_temp[j]
        y_temp[i] = (RHS[0,i] - sum)/R[i,i]

    # Now have the system R^T*x = y
    # R^T is upper triangular, so can use back substitution

    RT = R.transpose()

    for i in range(0, m)[::-1]:
        sum = 0
        for j in range(i, m):
            sum += RT[i, j]*x_coeffs[j]
        x_coeffs[i] = (y_temp[i]-sum)/RT[i,i]

    x_model = np.linspace(x[0], x[-1], 1000)
    model = np.zeros(1000)
    for i, c in enumerate(x_coeffs):
        model += c*x_model**i

    return x_model, model


# data set 1
n = 30
start = -2
stop = 2
x_1 = np.linspace(start, stop, n)
eps = 1
np.random.seed(1)
r = np.random.random(n) * eps
y_1 = x_1*(np.cos(r+0.5*x_1**3) + np.sin(0.5*x_1**3))

# data set 2
n = 30
start = -2
stop = 2
x_2 = np.linspace(start, stop, n)
eps = 1
np.random.seed(1)
r = np.random.random(n) * eps
y_2 = 4*x_2**5 - 5*x_2**4 - 20*x_2**3 + 10*x_2**2 + 40*x_2 + 10 + r

plt.figure()
x, m_3 = ex1(x_1, y_1, 3)
x, m_8 = ex1(x_1, y_1, 8)
plt.plot(x_1, y_1, "or")
plt.plot(x, m_3, "b")
plt.plot(x, m_8, "g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("QR Factorization, first data set")
plt.legend(["Data", "m = 3", "m = 8"])

plt.figure()
x, m_3 = ex1(x_2, y_2, 3)
x, m_8 = ex1(x_2, y_2, 8)
plt.plot(x_2, y_2, "or")
plt.plot(x, m_3, "b")
plt.plot(x, m_8, "g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("QR Factorization, second data set")
plt.legend(["Data", "m = 3", "m = 8"])

plt.figure()
x, m_3 = ex2(x_1, y_1, 3)
x, m_8 = ex2(x_1, y_1, 8)
plt.plot(x_1, y_1, "or")
plt.plot(x, m_3, "b")
plt.plot(x, m_8, "g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cholesky Factorization, first data set")
plt.legend(["Data", "m = 3", "m = 8"])

plt.figure()
x, m_3 = ex2(x_2, y_2, 3)
x, m_8 = ex2(x_2, y_2, 8)
plt.plot(x_2, y_2, "or")
plt.plot(x, m_3, "b")
plt.plot(x, m_8, "g")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cholesky Factorization, second data set")
plt.legend(["Data", "m = 3", "m = 8"])
plt.show()
