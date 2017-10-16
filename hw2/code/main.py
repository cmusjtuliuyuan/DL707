import optparse
from RBM import *
from Autoencoder import *

optparser = optparse.OptionParser()
optparser.add_option(
    "-p", "--problem", default="a",
    type='str', help="Choose which subproblem to solve(a,b,c,d,e,f,g)"
)

def main():
    opts = optparser.parse_args()[0]

    if opts.problem == 'a':
        problem_a_b(k=1)

    if opts.problem == 'b':
        problem_a_b(k=5)
        problem_a_b(k=10)
        problem_a_b(k=20)

    if opts.problem == 'c':
        problem_c()

    if opts.problem == 'd':
        problem_d()

    if opts.problem == 'e':
        problem_e_f(Denoise = False)

    if opts.problem == 'f':
        problem_e_f(Denoise = True)

    if opts.problem == 'g': 
        problem_g(hidden_dim = 50)
        problem_g(hidden_dim = 100)
        problem_g(hidden_dim = 200)
        problem_g(hidden_dim = 500)


if __name__ == '__main__':
    main()