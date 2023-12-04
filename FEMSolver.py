import numpy as np
import matplotlib.pyplot as plt
# Jakob Long, JRL4725

# returns original function
def f(x,t):
    return(np.pi*np.exp(-t)*np.sin(np.pi*x))

# returns analytical solution
def u_a(x,t):
    return(np.exp(-t)*np.sin(np.pi*x))

# returns u(x,0) = sin(pi*x)
def u_boundary(x):
    return(np.sin(np.pi*x))

# initializes all of our necessary matrices & step sizes 
def initializations(N,n):
    ts = np.linspace(0,1,n+1)
    xi = np.linspace(0,1,N)
    dt = 1/n
    h = xi[1] - xi[0]
    Kg = np.zeros((N,N))
    Mg = np.zeros((N,N))
    Fg = np.zeros((N,N+1))
    Kloc = np.zeros((2,2))
    Mloc = np.zeros((2,2))
    local2global_map = np.vstack((np.linspace(0, N-2, N-1), np.linspace(1, N-1, N-1))).T
    return ts, xi, dt, h, Kg, Fg, Mg, Kloc, Mloc, local2global_map

# Creates our phi test functions
def phi_functions(h):
    phi_1 = lambda eta: (1-eta)/2
    phi_2 = lambda eta: (1+eta)/2
    
    dphi = np.array([-1/2,1/2])
    
    dx_deta = h/2
    deta_dx = 2/h
    
    #Quadrature weights for two points
    weight_1 = -1/np.sqrt(3)
    weight_2 = 1/np.sqrt(3)
    
    phi_eta = np.array([[phi_1(weight_1), phi_1(weight_2)],
                       [phi_2(weight_1), phi_2(weight_2)]])
    
    return phi_eta, deta_dx, dx_deta, dphi

def stiff_force_matrices(N,phi_eta,local2global_map,Kg,Fg,Mg,Kloc,Mloc,h,deta_dx,dphi,dx_deta,ts):
    for k in range(N-1):
        for l in range(1):
            # Construction of local matrices
            for m in range(1):
                Mloc[l,m] = ((phi_eta[0,0] * phi_eta[m,0] + phi_eta[l,1] * phi_eta[m,1]) * h)
                Kloc[l,m] = dphi[l] * deta_dx * dphi[m] *deta_dx*dx_deta**2
        # conversion of local to global
        for l in range(1):
            global_node = int(local2global_map[k,l])
            global_node = global_node + 1
            # construction of global matrices
            for m in range(1):
                global_node2 = int(local2global_map[k,m])
                global_node2 = global_node2 + 1
                Kg[global_node,global_node2] = Kg[global_node,global_node2] + Kloc[l,m]
                Mg[global_node,global_node2] = Mg[global_node,global_node2] + Mloc[l,m]
        # Builds force global matrice
        value_to_assign = np.array(-((f(-1 / np.sqrt(3), ts) * phi_eta[0, 0] + f(1 / np.sqrt(3), ts) * phi_eta[0, 1]) * (1 / 8)))
        Fg[k,:-1] = value_to_assign[:len(Fg[k,:-1])]
    # applies boundary conditions to mass matrix
    Mg[0,:] = 0
    Mg[:,0] = 0
    Mg[N-1,:] = 0
    Mg[:,N-1] = 0
    Mg[0,0] = 1
    Mg[N-1,N-1] = 1
    
    return Kg,Mg,Fg

# applies & constructs boundary conditions
def boundary_conditions(xi,N,n):
    u = np.zeros((N,n))
    u[:,1] = u_boundary(xi)
    
    bounds = [0,0]
    dirichlet_bc = np.eye(N)
    dirichlet_bc[0,0] = bounds[0]
    dirichlet_bc[N-1,N-1] = bounds[1]
    
    return u, dirichlet_bc

# performs forward euler
def forward_euler(u,dt,MK,InvMg,Fg,dbc,n):
    for t in range(n-1):
        u[:,t+1] = u[:,t] - dt*np.dot(MK,u[:,t]) + dt*np.dot(InvMg,Fg[:,t])
        u[:,t+1] = dbc @ u[:,t+1]
    return u

# performs backward euler
def backward_euler(u,dt,invB,Mg,Fg,dbc,n):
    for t in range(n-1):
        u[:,t+1] = (1/dt)*invB*Mg*u[:,t] + invB*Fg[:,t]
        u[:,t+1] = dbc*u[:,t+1]
    return u

def main():
    N = int(input("Whats the number of Nodes: "))
    n = int(input("Whats the number of time steps:  "))

    
    ts,xi,dt,h,Kg,Fg,Mg,Kloc,Mloc,local2global_map = initializations(N,n)
    phi_eta,dx_deta,deta_dx,dphi = phi_functions(h)
    Kg, Mg,Fg = stiff_force_matrices(N,phi_eta,local2global_map,Kg,Fg,Mg,Kloc,Mloc,h,deta_dx,dphi,dx_deta,ts)
    InvMg = np.linalg.inv(Mg)
    MK = np.matmul(InvMg,Kg)
    B = ((1/dt) * Mg) + Kg 
    InvB = np.linalg.inv(B)
    u, dbc = boundary_conditions(xi,N,n)
    print(u.shape)
    print(MK.shape)
    print(InvMg.shape)
    print(Fg.shape)
    
    while True:
        method = str(input("Please type FE for Forward Euler or Please type BE for Backward Euler: ").upper())
        if method == 'FE':
            u = forward_euler(u,dt,MK,InvMg,Fg,dbc,n)
            break
        elif method == 'BE':
            u = backward_euler(u,dt,InvB,Mg,Fg,dbc,n)
            break
        else:
            print("Please select FE or BE.")
    
    
    
        
if __name__ == "__main__":
  main()