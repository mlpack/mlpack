#include "fastlib/fastlib.h"
#include "obj_grad_hess_class.h"  

int main(int argc, char *argv[]) {

	fx_module *fx_root = fx_init(argc, argv, NULL);

	index_t num_points_t = fx_param_int_req(fx_root, "number of points for t");

	ObjectGradientHessian my_objective;
	my_objective.Init(num_points_t);

	
	double dummy_res;
	my_objective.ApproxBetafnNominator(num_points_t , &dummy_res)

	fx_done(fx_root);
}