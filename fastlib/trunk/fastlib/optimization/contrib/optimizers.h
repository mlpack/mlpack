/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file optimizers.h
 *
 * Declares classes for two types of optimizer
 *
 */

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "../../fastlib.h"

const fx_entry_doc opt_entries[] = {
  {"method", FX_PARAM, FX_STR, NULL,
   " The method used to optimize.\n"},
  {"param_space_dim", FX_RESERVED, FX_INT, NULL,
   " The dimension of the parameter space.\n"},
  {"init_opt", FX_TIMER, FX_CUSTOM, NULL,
   " The time taken to initialize the optimizer.\n"},
  {"get_init_pt", FX_TIMER, FX_CUSTOM, NULL,
   " The time taken to generate the initial point.\n"},
  {"get_init_pts", FX_TIMER, FX_CUSTOM, NULL,
   " The time taken to obtain the 'dim+1' points required"
   " to use the NelderMead optimizer.\n"},
  {"optimizing", FX_TIMER, FX_CUSTOM, NULL,
   " The time taken to get to the optimal value.\n"},
  {"tolerance", FX_PARAM, FX_DOUBLE, NULL,
   " The tolerance value for the parameters"
   " (defaults to 1.0e-5).\n"},
  {"MAX_FUNC_EVAL", FX_PARAM, FX_INT, NULL,
   " The maximum number of function evaluations"
   " allowed to the NelderMead optimizer (defaults"
   " to 50000).\n"},
  {"func_evals", FX_RESULT, FX_INT, NULL,
   " The number of function evaluations taken by the algorithm.\n"},
  {"EPSILON", FX_PARAM, FX_DOUBLE, NULL,
   " Value of epsilon.\n"},
  {"TOLERANCE", FX_PARAM, FX_DOUBLE, NULL,
   " Tolerance for the minimum movement for the parameter value.\n"},
  {"gtol", FX_PARAM, FX_DOUBLE, NULL,
   " Tolerance value for the gradient of the function.\n"},
  {"MAX_STEP_SIZE", FX_PARAM, FX_DOUBLE, NULL,
   " The maximum step size in the direction of the gradient.\n"},
  {"MAX_ITERS", FX_PARAM, FX_INT, NULL,
   " The maximum number of iterations allowed to the function.\n"},
  {"iters", FX_RESULT, FX_INT, NULL,
   " The number of iterations the algorithm actually went through"
   " before reaching the apparent optimum.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc opt_doc = {
  opt_entries, NULL,
  " This file containes two optimizers.\n"
};

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
 * index_t number_of_function_evaluations;
 * struct datanode *opt_module = fx_submodule(NULL,"NelderMead","opt_module");
 * Matrix data;
 * index_t dim_param_space;
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
  index_t dimension_;
  Matrix data_;
  long double (*func_ptr_)(Vector&, const Matrix&);
  datanode *opt_module_;

 public:

  NelderMead() {
  }

  ~NelderMead() {
  }

  void Init(long double (*fun)(Vector&, const Matrix&),
	    Matrix& data, datanode *opt_module) {
    data_.Copy(data);
    func_ptr_ = fun;
    opt_module_ = opt_module;
    dimension_ = fx_param_int_req(opt_module_, "param_space_dim");
  }

  const Matrix& data() {
    return data_;
  }

  index_t dimension() {
    return dimension_;
  }

  void Eval(double **pts);
  long double ModSimplex_(double **pts, long double *y,
			  double *psum, index_t ihi, float fac); 
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
 * index_t number_of_iters;
 * struct datanode *opt_module = fx_submodule(NULL,"QuasiNewton","opt_module");
 * Matrix data;
 * index_t dim_param_space;
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
  index_t dimension_;
  Matrix data_;
  long double (*func_ptr_)(Vector&, const Matrix&, Vector*);
  datanode *opt_module_;

 public:

  QuasiNewton(){
  }

  ~QuasiNewton(){ 
  }

  void Init(long double (*fun)(Vector&, const Matrix&, Vector*),
	    Matrix& data, datanode *opt_module){
	  
    data_.Copy(data);
    func_ptr_ = fun;
    opt_module_ = opt_module;
    dimension_ = fx_param_int_req(opt_module_, "param_space_dim");
  }

  const Matrix data() {
    return data_;
  }

  index_t dimension() {
    return dimension_;
  }

  void Eval(double *pt);
  void LineSearch_(Vector pold, long double fold, Vector *grad,
		   Vector *xi, Vector *pnew, long double *f_min,
		   long double maximum_step_length);
};

#endif
