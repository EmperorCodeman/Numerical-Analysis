import numpy as np

#questions 5(a), 5(c) and questions 7(a), 7(c)

# Exercise 1 (a) 
a_A = np.array([\
     [3, -1,  1],
     [3,  6,  2],
     [3,  3,  7]
     ])

a_C = np.array([1,  0,  4]).transpose()

# Exercise 1 (c)
c_A = np.array([\
    [10,  5,  0,   0],
    [5,  10, -4,   0],
    [0,  -4,  8,  -1],
    [0,   0,  -1,  5],
    ])

c_C = np.array([6, 25, -11, -11])


def system_to_iterative_matrix(system):
    #   Solve the system for each var. Not row echelon form. Solve each var as a linear combination of the other vars. Leaving the diagonal as zeros   
    system *= -1 #  Move the terms to the other side of the equation
    system = (system.transpose() / system.diagonal()).transpose() # Divide by each rows respective vars coefficients being solved for
    np.fill_diagonal(system, 0) #   The diagonal is 0 because the respective term is on the other side of the equation
    return system

def is_diagonally_dominate(fixed_point_matrix):
    "convergence is guaranteed"
    pass

def solve_system_of_equations(system_coefficient_matrix, system_constants, ERROR_TOLERANCE=10e-3, algorithm="Gauss-Seidel", problem_name=""):
    #   System should be a square matrix. a_1*x_1 ... a_n*x_n = constants

    def get_error(current_step, last_step):
        return abs(current_step - last_step).max()
        
    B = system_to_iterative_matrix(system_coefficient_matrix)
    last_step = np.zeros_like(system_constants)
    current_step = B @ last_step + system_constants 
    current_error = get_error(current_step, last_step) 
    log = "\n\n\n\n\t\t\t" + problem_name + "\n\t\t\tAlgorithm: " + algorithm
    i = 1

    def update_log(i, current_error, current_step):
        return "\n\n\tIteration: " + str(i) + "\nError: " + str(current_error) + "\nEstimated Roots: " + str(current_step)

    while current_error > ERROR_TOLERANCE:
        last_step = np.copy(current_step)
        if algorithm == "Gauss-Seidel":
            for ii, b in enumerate(B):
                current_step[ii] = b @ current_step + system_constants[ii]
        else: # Jacobi Method
            current_step = B @ last_step + system_constants 
        current_error = get_error(current_step, last_step) 
        log += update_log(i, current_error, current_step)    
        i += 1

    print(log)
    return current_step, log

solve_system_of_equations(a_A, a_C, algorithm="Jacobi", problem_name="Problem 1 (a)")
solve_system_of_equations(c_A, c_C, algorithm="Jacobi", problem_name="Problem 1 (c)")
solve_system_of_equations(a_A, a_C, algorithm="Gauss-Seidel", problem_name="Problem 1 (a)")
solve_system_of_equations(c_A, c_C, algorithm="Gauss-Seidel", problem_name="Problem 1 (c)")

