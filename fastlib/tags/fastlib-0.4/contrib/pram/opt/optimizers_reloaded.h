/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file optimizers_reloaded.h
 *
 * Implements classes for two types of optimizer
 *
 */

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <fastlib/fastlib.h>

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
  //long double (*func_ptr_)(Vector&, const Matrix&, Vector*, Vector*, Vector*) ;
  datanode *opt_module_;

 public:

  QuasiNewton(){
  }

  ~QuasiNewton(){ 
  }

  void Init(long double (*fun)(Vector&, const Matrix&, Vector*),
	    //long double (*fun)(Vector&, const Matrix&, Vector*, Vector*, Vector*),
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

/**
 * Normal Gradient Descent implemented here
 * documentation later
 *
 */

class GradientDescent {

 private:
  index_t dimension_;
  Matrix data_;
  long double (*func_ptr_)(Vector&, const Matrix&, Vector*);
  datanode *opt_module_;

 public:

  GradientDescent(){
  }

  ~GradientDescent(){ 
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
 
};

/**
 * Stochastic Gradient Descent implemented here
 * documentation later
 *
 */

class SGD {

 private:
  index_t dimension_;
  Matrix data_;
  long double (*func_ptr_)(Vector&, const Matrix&, Vector*);
  datanode *opt_module_;

 public:

  SGD(){
  }

  ~SGD(){ 
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

  void PermuteMatrix_(const Matrix& input, Matrix *output) {
  
    ArrayList<index_t> perm_array;
    index_t size = input.n_cols();
    Matrix perm_mat;

    perm_mat.Init(size, size);
    perm_mat.SetAll(0.0);

    math::MakeRandomPermutation(size, &perm_array);
    for(index_t i = 0; i < size; i++) {
      perm_mat.set(perm_array[i], i, 1.0);
    }

    la::MulInit(input, perm_mat, output);
    return;
  }
};


/**
 * Stochastic Meta Descent with a 
 * Single step model implemented here
 * documentation later
 *
 */


class SMD {

 private:
  index_t dimension_;
  Matrix data_;
  // This is the original way of calling the function
  long double (*func_ptr_)(Vector&, const Matrix&, Vector*);

  // This is for the multistep method in which we need the 
  // Hessian-vector product
  // long double (*func_ptr_)(Vector&, const Matrix&, Vector*, Vector&, Vector*);

  // But since the L2 function is pretty awesome, we need 
  // to make the calling a little diff
  // long double (*func_ptr_stoc_)(Vector&, const Matrix&, Vector*, index_t);
  datanode *opt_module_;

 public:

  SMD(){
  }

  ~SMD(){ 
  }

  void Init(long double (*fun)(Vector&, const Matrix&, Vector*), 
	    // long double (*fun)(Vector&, const Matrix&, Vector*, Vector&, Vector*)
	   //long double (*fun_stoc)(Vector&, const Matrix&, Vector*, index_t),
	    Matrix& data, datanode *opt_module){
	  
    data_.Copy(data);

    // function pointer to the original function
    func_ptr_ = fun;

    // pointer to the broken up function used 
    // for stochastic optimization
    // func_ptr_stoc_ = fun_stoc;
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

  void PermuteMatrix_(const Matrix& input, Matrix *output) {
  
    ArrayList<index_t> perm_array;
    index_t size = input.n_cols();
    Matrix perm_mat;

    perm_mat.Init(size, size);
    perm_mat.SetAll(0.0);

    math::MakeRandomPermutation(size, &perm_array);
    for(index_t i = 0; i < size; i++) {
      perm_mat.set(perm_array[i], i, 1.0);
    }

    la::MulInit(input, perm_mat, output);
    return;
  }

  void HadamardOverwrite(Vector& a, Vector& b, Vector *prod) {
    DEBUG_SAME_SIZE(a.length(), b.length());
    DEBUG_SAME_SIZE(a.length(), prod->length());
    double *x = a.ptr();
    double *y = b.ptr();
    index_t length = a.length();
    double *z = prod->ptr();

    do {
      *z++ = (*x++)*(*y++);
    }while(--length);
  }

  void HadamardInit(Vector& a, Vector& b, Vector *prod) {
    DEBUG_SAME_SIZE(a.length(), b.length());
    prod->Init(a.length());
    HadamardOverwrite(a, b, prod);
  }

  void HadamardTo(Vector& a, Vector *b) {
    DEBUG_SAME_SIZE(a.length(), b->length());
    Vector c;
    HadamardInit(a, *b, &c);
    b->CopyValues(c);
  }
};


#endif
