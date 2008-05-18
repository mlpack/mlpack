This folder contains the implementation of two optimizer classes, namely the Nelder-Mead' polytope method and the BFGS form of the Quasi Newton method of optimization. The header file contains the implementation of these classes and the way to use them is to declare an object of that class and initialize them with the function to be optimized and the dataset with respect to which they are to be optimized. An example of use would be as following:

 double init_pts[d+1][d];
 index_t number_of_function_evaluations;
 struct datanode *opt_module = fx_submodule(NULL,"NelderMead");
 Matrix data;
 index_t dim_param_space;

 ...
 NelderMead opt;
 opt.Init(obj_function, data, dim_param_space, opt_module);
 ...
 opt.Eval(init_pts);
 // init_pts[0] contains the optimal point found

The QuasiNewton class can be used in the same way except here we have to just give a single initial point, and the function to be optimized takes input an additional vector in which the computed gradients would be passed by the function. So the function to be optimized would be of the form :
long double obj_func(Vector& opt_params, const Matrix& data, Vector *gradients);

