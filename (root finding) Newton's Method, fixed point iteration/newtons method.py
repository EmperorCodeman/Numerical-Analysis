import numpy as np
"""
    Newtons method is a better way of doing fixed point iteration, then using algebra to turn a function = 0, into a map that maps into itself, g(x) = x
    This is because we dont need to meet the suppositions to insure convergence. 
    Instead, simply, use the x intercept given by an arbitrary point's tangent line, as the next element of the series 
    The x intercept is given by, x intercept of x_1's tangent line = x_2 = f(x_1) - f(x_1) / f'(x_1)
    For convenance you can approximate f'(x) with the secant line 
    Convergence is proven given x is close enough to the root. 
    You may also keep one point on the secant and only replace it when the next member of the series has a opposite sign than the other point forming the secant. This insures that a root is between them
    Note: This program in not optimized at all. 
"""

# TODO EXAMINE HERE!!  
initial_x = 2 # If this value, is set to 2, then the secant method is convergent. However, with one, a interesting result occurs. Where, the series, approximately, alternates! Can you explain, this occurrence? Is it due to floating point arithmetic? Is the loop error, escaped for all similar problems, when, the bracket method is used? 
problem_5_a = [lambda x: x**3 - 2*x**2 - 5, lambda x: 3*x**2 - 4*x, initial_x, "Problem 5 a"]
problem_5_c = [lambda x: x - np.cos(x), lambda x: 1 + np.sin(x), 0, "Problem 5 c"]
#   A problem is [f(x), f'(x), starting x]
problems = [problem_5_a, problem_5_c]
TOL, I_LIMIT = 1.0e-6, 200
newtons_method = lambda f_of_x, f_prime_of_x, x: x - f_of_x(x)/f_prime_of_x(x)   
secant_method = lambda f_of_x, x_1, x_2: [x_2 - f_of_x(x_2) * (x_2 - x_1) / (f_of_x(x_2) - f_of_x(x_1)), x_2]
results = ["\n\t\tResults Summary"]

for problem in problems:
    #   Newton's Method
    i = 0
    initial_x = problem[2]
    print("\n\n\t" + problem[3].upper() + " Newton's Method")
    while abs(problem[0](problem[2])) > TOL: # Test is sufficiently close to the root
        i += 1
        # update, to next point, in the series.
        problem[2] = newtons_method(problem[0], problem[1], problem[2])
        print(str(i) + " th iteration, x_i " + str(problem[2]) + "\ty_i " + str(problem[0](problem[2])) )
        if i == I_LIMIT: break
    converged_readout = problem[3] + " converged with newtons method with " + str(i) + " iterations"
    not_converged_readout = problem[3] + ", failed to converge, to the root, with newton's method."
    results.append(converged_readout) if i < I_LIMIT else results.append(not_converged_readout) 

    #   Secant Method
    print("\n\t" + problem[3].upper() + " Secant Method")
    i = 0
    problem[2] = initial_x + 1
    x_1 = initial_x
    while abs(problem[0](problem[2])) > TOL: # Test is sufficiently close to the root
        i += 1
        # update, to next point, in the series.
        problem[2], x_1 = secant_method(problem[0], problem[2], x_1) 
        print(str(i) + " th iteration, x_i " + str(problem[2]) + "\ty_i " + str(problem[0](problem[2])) )
        if i == I_LIMIT: break
    converged_readout = problem[3] + " converged with secant method with " + str(i) + " iterations"
    not_converged_readout = problem[3] + ", failed to converge, to the root, with the secant method."
    results.append(converged_readout) if i < I_LIMIT else results.append(not_converged_readout) 

for result in results: print(result)
    
