import optparse
from Problem6 import *
from nn import Sigmoid, ReLU, Tanh
from nn import NN_3_layer, NN_4_layer, NN_4_BN_layer

optparser = optparse.OptionParser()
optparser.add_option(
    "-p", "--problem", default="a",
    type='str', help="Choose which subproblem to solve(a,b,c,d,e,f,g,h,i)"
)

def main():
	opts = optparser.parse_args()[0]

	if opts.problem == 'a':
		fig = plot_loss_average(info = 'cross_entropy_loss', ymax = 1, NLL=True)
		fig.savefig('problem_a.png')

	if opts.problem == 'b':
		fig = plot_loss_average(info = 'incorrect classification ratio', ymax = 0.5, NLL=False)
		fig.savefig('problem_b.png')

	if opts.problem == 'c':
		_, _, model = get_loss_one_time(NLL=True)
		fig = plot_visulizing_parameter(np.transpose(model.layer1.W[:-1,:]), 10)
		fig.savefig('problem_c.png')

	if opts.problem == 'd':
		fig = plot_problem_d_learning_rt('cross_entropy_loss', 2, NLL=True)
		fig.savefig('problem_d_learning_rt_cross_entropy.png')

		fig = plot_problem_d_learning_rt('incorrect classification ratio', 1, NLL=False)
		fig.savefig('problem_d_learning_rt_IC.png')

		fig = plot_problem_d_momentum('cross_entropy_loss', 3, NLL=True)
		fig.savefig('problem_d_momentum_cross_entropy.png')
		fig = plot_problem_d_momentum('incorrect classification ratio', 1, NLL=False)
		fig.savefig('problem_d_momentum_IC.png')

	if opts.problem == 'e':
		fig = plot_problem_e()
		fig.savefig('problem_e.png')

	if opts.problem == 'f':
		#train_loss_NLL 0.0302798907306 valid_loss_NLL 0.274775637271 test_loss_NLL 0.325125106534
		#train_loss_IC 0.0 valid_loss_IC 0.084 test_loss_IC 0.0923333333333
		fig = plot_problem_f_l2_reg()
		fig.savefig('problem_f_find_l2.png')
		model = NN_3_layer()
		fig = plot_problem_f_or_g_result(model, 180, 0.01, 0, 0.001)
		fig.savefig('problem_f_visualization.png')

	if opts.problem == 'g':
		#train_loss_NLL 0.324571271092 valid_loss_NLL 0.499887242168 test_loss_NLL 0.552144295303
		#train_loss_IC 0.101 valid_loss_IC 0.149 test_loss_IC 0.166666666667
		fig = plot_problem_f_l2_reg()
		fig.savefig('problem_f_find_l2.png')
		model = NN_3_layer()
		fig = plot_problem_f_or_g_result(model, 180, 0.01, 0, 0.001)
		fig.savefig('problem_f_visualization.png')

	if opts.problem == 'h':
		model = NN_4_BN_layer()
		#Epoch num: 7
		#train_loss_NLL 0.115590739458 valid_loss_NLL 0.359999605125 test_loss_NLL 0.528242543282
		#train_loss_IC 0.036 valid_loss_IC 0.118 test_loss_IC 0.138
		for i in range(30):

		    print 'Epoch num:', i
		    # Train the model
		    train_or_evaluate_epoch(model, train_data, Train = True,
		                         learning_rt = 0.1, momentum = 0, NLL = True, alpha = 0)

		    train_loss_NLL = train_or_evaluate_epoch(model, train_data, Train = False, NLL = True)
		    valid_loss_NLL = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = True)
		    test_loss_NLL= train_or_evaluate_epoch(model, test_data, Train = False, NLL = True)
		    train_loss_IC = train_or_evaluate_epoch(model, train_data, Train = False, NLL = False)
		    valid_loss_IC = train_or_evaluate_epoch(model, valid_data, Train = False, NLL = False)
		    test_loss_IC= train_or_evaluate_epoch(model, test_data, Train = False, NLL = False)
		    print 'train_loss_NLL', train_loss_NLL, 'valid_loss_NLL', valid_loss_NLL, 'test_loss_NLL', test_loss_NLL
		    print 'train_loss_IC', train_loss_IC, 'valid_loss_IC', valid_loss_IC, 'test_loss_IC', test_loss_IC
	
	if opts.problem == 'i':
		fig = plot_problem_i()
		fig.savefig('problem_i.png')

if __name__ == '__main__':
    main()