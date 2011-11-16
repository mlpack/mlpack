/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file optimizers.hpp
 *
 * Declares classes for two types of optimizer
 *
 */
#ifndef __MLPACK_METHODS_MOG_OPTIMIZERS_HPP
#define __MLPACK_METHODS_MOG_OPTIMIZERS_HPP

#include <mlpack/core.h>

PARAM_STRING("method", "The method used to optimize", "opt", "");

PARAM_INT_REQ("param_space_dim", "The dimension of the parameter space.", "opt");
PARAM_INT("MAX_FUNC_EVAL", "The maximum number of function evaluations\
 allowed to the NelderMead optimizer (defaults to 50000)", "opt", 50000);

PARAM_INT("func_evals", "The number of function evaluations taken by the algorithm", "opt", 0);
PARAM_INT("MAX_ITERS", "The maximum number of iterations allowed to the function", "opt", 200);
PARAM_INT("iters", "The number of iterations the algorithm actually went through", "opt", 0);

PARAM_DOUBLE("EPSILON", "Value of epsilon.", "opt", 3.0e-8);
PARAM_DOUBLE("TOLERANCE", "Tolerance for the minimum movement for the parameter value.", "opt", 1.0e-5);
PARAM_DOUBLE("gtol", "Tolerance value for the gradient of the function", "opt", 1.0e-7);
PARAM_DOUBLE("MAX_STEP_SIZE", "The maximum step size in the direction of the gradient.", "opt", 100.0);
PARAM_DOUBLE("tolerance", "Undocumented parameter", "opt", 1.0e-5);

PARAM_MODULE("opt", "This file contains two optimizers.");

namespace mlpack {
namespace gmm {

/**
 * An optimizer using the Nelder Mead method,
 * also known as the polytope or the simplex
 * method.
 *
 * It does multivariate minimization of an
 * objective function. If it is optimizing in
 * 'd' dimensions, it would require 'd+1'
 * starting points.
 *
 * Example use:
 *
 * @code
 * double init_pts[d+1][d];
 * size_t number_of_function_evaluations;
 * struct datanode *opt_module = fx_submodule(NULL,"NelderMead","opt_module");
 * Matrix data;
 * size_t dim_param_space;
 *
 * ...
 * NelderMead opt;
 * opt.Init(obj_function, data, dim_param_space, opt_module);
 * ...
 * opt.Eval(init_pts);
 * // init_pts[0] contaings the optimal point found
 * @endcode
 *
 */
class NelderMead {
 private:
  size_t dimension_;
  arma::mat data_;
  long double (*func_ptr_)(const arma::vec&, const arma::mat&);

 public:
  NelderMead() { }
  ~NelderMead() { }

  void Init(long double (*fun)(const arma::vec&, const arma::mat&),
            arma::mat& data) {
    data_ = data;
    func_ptr_ = fun;
    dimension_ = mlpack::CLI::GetParam<int>("opt/param_space_dim");
  }

  const arma::mat& data() {
    return data_;
  }

  size_t dimension() {
    return dimension_;
  }

  void Eval(arma::mat& pts);
  long double ModSimplex_(arma::mat& pts, arma::vec& y,
                          arma::vec& psum, size_t ihi, float fac);
};

/**
 * An optimizer using the Quasi Newton method,
 * also known as the variable metrics
 * method.
 *
 * It does multivariate minimization of an
 * objective function using only the function
 * value and the gradients.
 *
 * Example use:
 *
 * @code
 * double init_pt[d];
 * size_t number_of_iters;
 * struct datanode *opt_module = fx_submodule(NULL,"QuasiNewton","opt_module");
 * Matrix data;
 * size_t dim_param_space;
 *
 * ...
 * QuasiNewton opt;
 * opt.Init(obj_function, data, dim_param_space, opt_module);
 * ...
 * opt.Eval(init_pt);
 * // init_pt contains the optimal point found
 * @endcode
 *
 */
class QuasiNewton {
 private:
  size_t dimension_;
  arma::mat data_;
  long double (*func_ptr_)(const arma::vec&, const arma::mat&, arma::vec&);

 public:
  QuasiNewton(){ }
  ~QuasiNewton(){ }

  void Init(long double (*fun)(const arma::vec&, const arma::mat&, arma::vec&),
            arma::mat& data) {
    data_ = data;
    func_ptr_ = fun;
    dimension_ = mlpack::CLI::GetParam<int>("opt/param_space_dim");
  }

  const arma::mat& data() {
    return data_;
  }

  size_t dimension() {
    return dimension_;
  }

  void Eval(arma::vec& pt);
  void LineSearch_(arma::vec& pold, long double fold, arma::vec& grad,
                   arma::vec& xi, arma::vec& pnew, long double& f_min,
                   long double maximum_step_length);
};

}; // namespace gmm
}; // namespace mlpack

#endif // __MLPACK_METHODS_MOG_OPTIMIZERS_HPP
