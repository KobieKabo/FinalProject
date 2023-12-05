# COE 352 Project 2 - Finite Element Solver
### _Jakob Long_

### Prereques For Running the Code
- pip install numpy, and matplotlib

## Project Overview
For this FEM solver, we addressed the heat equation that had the following form: u<sub>t</sub> - u<sub>xx</sub> = f(x,t), where (x,t) ∈ (0,1) x (0,1). With the following initial & Dirichlet boundary condiitons:

u(x,0) = sin(𝜋𝑥)
u(0,t) = u(1,t) = 0

Additionally, we were given f(x,t) = (𝜋<sup>2</sup> - 1)e<sup>-t</sup>sin(𝜋𝑥) & u(x,t) = e<sup>-t</sup>sin(𝜋𝑥)
With our inital problem state now properly defined, we can begin to explore the possible solution. 

To address this PDE, we used a 1-D galerkin code, using a generalized number of N nodes, but N = 11 for our examples below. Additionally, the 1-D lagrangian basis functions were used.
### Running the Code
First in the command line:
```
python3 .\FEMSolver.py
```

You will then get the following prompts that require a user input.
```
Enter the number of Nodes: 11
Enter the number of time steps: 551
Please type FE for Forward Euler or Please type BE for Backward Euler: FE
```
The M,K,F & U_final matrices will then  print, along with the plot output. Where U_final will depend on the method chosen: Forward Euler (FE) or Backward Euler (BE).

```

M:  (11, 11)
 [[1.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.        ]
 [0.         0.13333333 0.03333333 0.         0.         0.
  0.         0.         0.         0.         0.        ]
 [0.         0.03333333 0.13333333 0.03333333 0.         0.
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.03333333 0.13333333 0.03333333 0.
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.03333333 0.13333333 0.03333333
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.03333333 0.13333333
  0.03333333 0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.03333333
  0.13333333 0.03333333 0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.03333333 0.13333333 0.03333333 0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.03333333 0.13333333 0.03333333 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.03333333 0.13333333 0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         1.        ]]

K:  (11, 11)
 [[ 10. -10.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
 [-10.  20. -10.   0.   0.   0.   0.   0.   0.   0.   0.]
 [  0. -10.  20. -10.   0.   0.   0.   0.   0.   0.   0.]
 [  0.   0. -10.  20. -10.   0.   0.   0.   0.   0.   0.]
 [  0.   0.   0. -10.  20. -10.   0.   0.   0.   0.   0.]
 [  0.   0.   0.   0. -10.  20. -10.   0.   0.   0.   0.]
 [  0.   0.   0.   0.   0. -10.  20. -10.   0.   0.   0.]
 [  0.   0.   0.   0.   0.   0. -10.  20. -10.   0.   0.]
 [  0.   0.   0.   0.   0.   0.   0. -10.  20. -10.   0.]
 [  0.   0.   0.   0.   0.   0.   0.   0. -10.  20. -10.]
 [  0.   0.   0.   0.   0.   0.   0.   0.   0. -10.  10.]]

F:  (11, 552)
 [[0.62130202 0.62017545 0.61905093 ... 0.22939538 0.22897943 0.22856424]
 [0.62130202 0.62017545 0.61905093 ... 0.22939538 0.22897943 0.22856424]
 [0.62130202 0.62017545 0.61905093 ... 0.22939538 0.22897943 0.22856424]
 ...
 [0.62130202 0.62017545 0.61905093 ... 0.22939538 0.22897943 0.22856424]
 [0.62130202 0.62017545 0.61905093 ... 0.22939538 0.22897943 0.22856424]
 [0.         0.         0.         ... 0.         0.         0.        ]]

U:  (11, 552)
 [[0.00000000e+00 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
  0.00000000e+00 0.00000000e+00]
 [3.09016994e-01 3.13375246e-01 3.16320787e-01 ... 1.26403670e-01
  1.26174514e-01 1.25945774e-01]
 [5.87785252e-01 5.87710574e-01 5.88438918e-01 ... 2.27427021e-01
  2.27014726e-01 2.26603177e-01]
 ...
 [5.87785252e-01 5.87710574e-01 5.88438918e-01 ... 2.27427021e-01
  2.27014726e-01 2.26603177e-01]
 [3.09016994e-01 3.13375246e-01 3.16320787e-01 ... 1.26403670e-01
  1.26174514e-01 1.25945774e-01]
 [1.22464680e-16 0.00000000e+00 0.00000000e+00 ... 0.00000000e+00
  0.00000000e+00 0.00000000e+00]]
```
### Forward Euler Method

Graph of the stable Forward Euler method at n = 551


![alt text](https://raw.githubusercontent.com/KobieKabo/FinalProject/Main/FE_n551.png?raw=true)


Graph of when the stability of the Forward Euler method begins to unravel at n = 276. By lowering n, we're increasing our time step size.


![alt text](https://raw.githubusercontent.com/KobieKabo/FinalProject/Main/FE_n276.png?raw=true)

### Backward Euler Method

Graph of the stable Backward Euler method at n = 551


![alt text](https://raw.githubusercontent.com/KobieKabo/FinalProject/Main/BE_n551.png?raw=true)


Graph of the stable Backward Euler method at n = 276, where the accuracy will continue to decrease as we increase our time step size, or decrease n.

![alt text](https://raw.githubusercontent.com/KobieKabo/FinalProject/Main/BE_n276.png?raw=true)

## Project Questions

1. **At what dt does instability occur and how does the solution change as the number of nodes decreases?**

The 1D Galerkin method used, is unstable between dt's of 1/551 and 1/600. As the number of nodes decreases, the solution becomes less accurate and does not align with the analytical solution very well. 

2. **What happens as the timestep becomes greater than or equal to the spatial step size?** 

As the spatial step size becomes greater than or equal to the timestep, more information is forward propagating spatially than information dependent on time is, so the model is going to be inaccuarate since spatial accuracy is lost.





