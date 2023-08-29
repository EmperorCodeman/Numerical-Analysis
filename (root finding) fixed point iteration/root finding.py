import numpy as np
"""
    https://github.com/EmperorCodeman
    http://olympusstudios.club/
    License: Open for Use with Accreditation: 

    Given a function we seek to find its roots. g(x) = ... 
    To accomplish this, we add the input to the function to make a new function, h(x). The root of f(x), is the fixed point of h(x). 
    g(x_1) = 0, h(x_1) = x_1 = g(x_1) + x_1
    This new function, h(x), allows for the use of the fixed point algorithm.

    Fixed Point Algorithm:
        Given: A function maps into itself over a interval [a,b]; {x in [a,b] | g(x) in [a,b]},
            AND |g'(x)| <= 1 over [a,b]
        We have, by theorem 1; (Proof in code folder, built upon central limit theorem in calculus) 
            The limit, as n goes to infinity, for the series, x_n = h(x_(n-1)), ... , x_1 = h(x_0)
            1. Is ALWAYS! convergent to a root
            2. Their is one, and only one root in g(x) over [a,b]  
        
        In practice, we have a generic root finding problem as such;
            f(x) = 0 
        Before we would solve this equation for x, by factoring...
        Now, we can use this equation, in a new way. We can manipulate it into equality with x. Called fixed point form. There are a multitude of ways to arrive at a equality with x through manipulation. Some paths lead to undefined outcomes, or are not convergent
        Notice the new equation, can be viewed as a level of a function being set to equal itself. 
            a*f(x)/m(x) + h(x) = x = l(x)
        Thus we use skills to manipulate our equation to form a function that is convergent and easy. 
        This is done by getting a derivative that is as small as possible and a map that maps into itself  
            
        We then iteratively solve for x to achieve the root without factoring. 
"""
NUMERICAL_STABILITY_LIMIT = 10e6    #   The data structures on your computer cannot model numbers that are too large in value. The algorithm, is certainly, or almost certainly not convergent for any case exceeding these bounds 

#   Global Functions and Classes
class root_finding_problem:
    def __init__(self, name, map, domain, error_tolerance):
        self.name = name
        self.map = map 
        self.domain = domain    # Arbitrary starting estimate of the root, that, is within the bounds of the domain; permitted by theorem 1. Thus allowing, theorem 1, to insure convergence to the root!  
        self.error_tolerance = error_tolerance  # This is, the closeness to the root that is required. Expressed as, error tolerance. Because, we dont know the root! I, use the step's improvement as a pseudo error. Since, it follows that a convergent solution, must decrease, the abs difference between p_i and p_i-1, with each iteration.  

def map_domain_to_range(map, x):
    """
        Needs refactored to proper linear algebra, none the less, currently its decent
        To expand to multiple dimensions, x should be changed to vector, and recursion should be used. Then a nested loop can be used, such that each term is looped through, for each variable, in each term.
    """
    output = 0
    #   Process composition of functions. e(w(...u(x))) = x_
    composed_vector = []
    for function in map[1]: # TODO add recursion for elaborate composition of functions
        composed_vector.append( function(x) if function else x )
    for i, term in enumerate( map[0].transpose() ):
        term_coefficient, composites_index = term[0], term[1]
        output += term_coefficient * composed_vector[i]**composites_index # Summat each term's value given a value of x, thus yielding the transformed vector ouput 
    return output, abs(x - output)

def readout(roots_estimate, iterations_pseudo_error, iteration_count):
    print( str(iteration_count) + "th iteration: Roots estimate is " + str(roots_estimate) \
          + ".\t With a pseudo error of " + str(iterations_pseudo_error) ) 


#   Input to the program is the root problem class. Construct objects for each problem
problem_7_map = np.array([ 
    [ 1], # First row is, the coefficients of each term in the polynomial. Exp. 3x^4 => 3
    [ .5]  # Second row is, the indices of each terms respective variable. Exp. 3x^4 => 4, or 2x^0 => 0 
    ])
problem_7_composition = [lambda x: 3*x**-2 + 3] # Composition of functions, allow, for each term to encapsulate n composite functions
problem_7 = root_finding_problem(name="Problem #7", map=(problem_7_map, problem_7_composition), domain=[1,2], error_tolerance=10e-10)

problem_15_map = np.array([ 
    [ 1], 
    [ 1]  
    ])
problem_15_composition = [lambda x: np.arccos(-x**2/10)]
problem_15 = root_finding_problem("Problem #15 Root One", (problem_15_map, problem_15_composition), [3,3], 10e-10)
problem_15_map = np.array([ 
    [ 1], 
    [ 1]  
    ])
problem_15_composition = [lambda x: -np.arccos(-x**2/10)] # cos(x) = cos(-x) => cos(x) = c with arccos(c) = -acrcos(c), thus we have two roots that are symmetric
problem_15_ = root_finding_problem("Problem #15 Root Two", (problem_15_map, problem_15_composition), [3,3], 10e-10)

problem_15_map = np.array([ 
    [ 1/3], 
    [ 1]  
    ])
problem_15_composition = [lambda x: (2*x**2-10*np.cos(x))/x] # notice, x is also symmetric here. Because negative and positive values produce the same output. Even function 
problem_15__ = root_finding_problem("Problem #15 Root Three", (problem_15_map, problem_15_composition), [-5,5], 10e-10)

problem_15_map = np.array([ 
    [ 1/3], 
    [ 1]  
    ])
problem_15_composition = [lambda x: (2*x**2-10*np.cos(x))/x] # notice, x is also symmetric here. Because negative and positive values produce the same output. ie, this is a, Even function. However, notice, that unlike the arccos roots, the input is negated to arrive at the symmetry, the output is not negated like before; this difference is because acrcos is an inverse function, thus the output is x and the output needs negated to get the other root.  
problem_15___ = root_finding_problem("Problem #15 Root Four", (problem_15_map, problem_15_composition), [1000,-5], 10e-10) # Additionally, notice that I did not simply negate the last root, as is possible. Instead I stepped through, to the root, by starting above it. Any large starting point will work. Because the slope of the functions limits are asymptotic to a constant slope that is less than one.  


problem_test_map = np.array([
    [1/3, 0, -1/3],
    [2,   0,    0]
])
problem_test_composition = [None, None, None]
problem_test = root_finding_problem("Problem #test", (problem_test_map, problem_test_composition), [-3, 3], 10e-10)

root_finding_problems = [problem_7, problem_15, problem_15_, problem_15__, problem_15___]

#   Main method. Could be encapsulated for export
for problem in root_finding_problems:
    print("\n\n\t" + problem.name + "\n")
    roots_estimate, iterations_pseudo_error  = map_domain_to_range(problem.map, problem.domain[0]) 
    i = 1 

    #   Attempt to, iterate, to the solution, using fixed point root induction 
    while iterations_pseudo_error < NUMERICAL_STABILITY_LIMIT and problem.error_tolerance < iterations_pseudo_error:
        readout(roots_estimate, iterations_pseudo_error, i)
        roots_estimate, iterations_pseudo_error  = map_domain_to_range(problem.map, roots_estimate) 
        i += 1

    if problem.error_tolerance < iterations_pseudo_error:
        print("Incapable of computing with reasonable numerical stability: Failed to satisfy Error Tolerance!\n\t" + problem.name, " Garbage Root: " + str(roots_estimate))
    else:
        print("The method was successful\n\t" + problem.name + " Approximate Root: " + str(roots_estimate))

