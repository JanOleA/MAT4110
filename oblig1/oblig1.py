import numpy as np
import matplotlib.pyplot as plt

def make_A(x, y, m):
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

def ex1(x, y, m):
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

    model = np.zeros(n)
    for i, c in enumerate(x_coeffs):
        model += c*x**i

    return model


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
m_3 = ex1(x_1, y_1, 3)
m_8 = ex1(x_1, y_1, 8)
plt.plot(x_1, y_1, "or")
plt.plot(x_1, m_3, "b")
plt.plot(x_1, m_8, "g")
plt.legend(["Data", "m = 3", "m = 8"])

plt.figure()
m_3 = ex1(x_2, y_2, 3)
m_8 = ex1(x_2, y_2, 8)
plt.plot(x_2, y_2, "or")
plt.plot(x_2, m_3, "b")
plt.plot(x_2, m_8, "g")
plt.legend(["Data", "m = 3", "m = 8"])
plt.show()
