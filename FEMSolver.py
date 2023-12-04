import numpy as np
import matplotlib.pyplot as plt

# Function to return the original function
def f(x, t):
    return ((np.pi ** 2 - 1) * np.exp(-t) * np.sin(np.pi * x))

# Function to return the analytical solution
def u_analytical(x, t):
    return (np.exp(-t) * np.sin(np.pi * x))

# Function to return u(x,0) = sin(pi*x)
def u_boundary(x):
    return (np.sin(np.pi * x))

# Initialization of necessary matrices & step sizes
def initializations(N, n):
    ts = np.linspace(0, 1, n + 1)
    xi = np.linspace(0, 1, N)
    dt = 1 / n
    h = xi[1] - xi[0]
    local2global_map = np.vstack((np.linspace(0, N - 2, N - 1), np.linspace(1, N - 1, N - 1))).T
    return ts, xi, dt, h, local2global_map

# Creates phi test functions
def phi_functions(h):
    phi_1 = lambda eta: (1 - eta) / 2
    phi_2 = lambda eta: (1 + eta) / 2

    dphi = np.array([-1 / 2, 1 / 2])

    dx_deta = h / 2
    deta_dx = 2 / h

    weight_1 = -1 / np.sqrt(3)
    weight_2 = 1 / np.sqrt(3)

    phi_eta = np.array([[phi_1(weight_1), phi_1(weight_2)],
                        [phi_2(weight_1), phi_2(weight_2)]])

    return phi_eta, deta_dx, dx_deta, dphi

# Constructs matrices for stiffness, force, and applies boundary conditions
def stiff_force_matrices(N,n,phi_eta, local2global_map, h, deta_dx, dphi, dx_deta, ts):
    Kloc = np.zeros((2, 2))
    Mloc = np.zeros((2, 2))
    Kg = np.zeros((N, N))
    Mg = np.zeros((N, N))
    Fg = np.zeros((N, n + 1))
    
    for k in range(N - 1):
        for l in range(2):
            for m in range(2):
                Mloc[l, m] = ((phi_eta[l, 0] * phi_eta[m, 0] + phi_eta[l, 1] * phi_eta[m, 1]) * h)
                Kloc[l, m] = dphi[l] * deta_dx * dphi[m] * deta_dx * dx_deta * 2

        for l in range(2):
            global_node = int(local2global_map[k, l])
            global_node = max(0, min(N - 1, global_node))

            for m in range(2):
                global_node2 = int(local2global_map[k, m])
                global_node2 = max(0, min(N - 1, global_node2))
                Kg[global_node, global_node2] += Kloc[l, m]
                Mg[global_node, global_node2] += Mloc[l, m]

        Fg[k, :] = -(f(-1 / np.sqrt(3), ts) * phi_eta[0, 0] + f(1 / np.sqrt(3), ts) * phi_eta[0, 1]) * (1 / 8)

    Mg[0, :] = 0
    Mg[:, 0] = 0
    Mg[N - 1, :] = 0
    Mg[:, N - 1] = 0
    Mg[0, 0] = 1
    Mg[N - 1, N - 1] = 1

    return Kg, Mg, Fg

# Applies & constructs boundary conditions
def boundary_conditions(xi, N, n):
    u = np.zeros((N, n + 1))
    u[:, 0] = u_boundary(xi)

    boundaries = [0, 0]
    dirichlet_bc = np.eye(N)
    dirichlet_bc[0, 0] = boundaries[0]
    dirichlet_bc[N - 1, N - 1] = boundaries[1]

    return u, dirichlet_bc

# Forward Euler method
def forward_euler(u, dt, MK, InvMg, Fg, dbc, n):
    for t in range(n - 1):
        u[:, t + 1] = u[:, t] - dt * MK.dot(u[:, t]) + dt * InvMg.dot(Fg[:, t])
        u[:, t + 1] = dbc.dot(u[:, t + 1])
    return u

# Backward Euler method
def backward_euler(u, dt, invB, Mg, Fg, dbc, n):
    for t in range(n):
        u[:, t + 1] = (1 / dt) * invB.dot(Mg.dot(u[:, t])) + invB.dot(Fg[:, t])
        u[:, t + 1] = dbc.dot(u[:, t + 1])
    return u

# Main function
def main():
    N = int(input("Enter the number of Nodes: "))
    n = int(input("Enter the number of time steps: "))

    ts, xi, dt, h, local2global_map = initializations(N, n)
    phi_eta, dx_deta, deta_dx, dphi = phi_functions(h)
    Kg, Mg, Fg = stiff_force_matrices(N,n,phi_eta, local2global_map, h, deta_dx, dphi, dx_deta, ts)
    InvMg = np.linalg.inv(Mg)
    MK = np.dot(InvMg, Kg)
    B = ((1 / dt) * Mg) + Kg
    InvB = np.linalg.inv(B)
    u, dbc = boundary_conditions(xi, N, n)

    while True:
        method = str(input("Please type FE for Forward Euler or Please type BE for Backward Euler: ").upper())
        if method == 'FE':
            u = forward_euler(u, dt, MK, InvMg, Fg, dbc, n)
            print(u)
            x = np.linspace(0, 1, N)
            xn = np.linspace(0, 1, 1000)
            u_a = u_analytical(xn, 1)
            plt.plot(xn, u_a, label='Exact Solution')
            plt.plot(x, u[:, n - 1], label='Forward Euler Approximation')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.legend()
            plt.show()
            break
        elif method == 'BE':
            u = backward_euler(u, dt, InvB, Mg, Fg, dbc, n)
            x = np.linspace(0, 1, N)
            xn = np.linspace(0, 1, 1000)
            u_a = u_analytical(xn, 1)
            plt.plot(xn, u_a, label='Exact Solution')
            plt.plot(x, u[:, n - 1], label='Forward Euler Approximation')
            plt.xlabel('x')
            plt.ylabel('u')
            plt.legend()
            plt.show()
            break
        else:
            print("Please select FE or BE.")

if __name__ == "__main__":
    main()
