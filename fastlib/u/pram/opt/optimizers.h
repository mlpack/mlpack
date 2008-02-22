/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file optimizers.h
 *
 * Implements classes for two types of optimizer
 *
 */

#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <fastlib/fastlib.h>

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
 * // init_pts[0] contains the optimal point found
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

  void Eval(double **pts) {

    index_t dim = dimension(), num_func_eval;
    index_t i, j, ihi, ilo, inhi,mpts = dim + 1;
    double sum, swap, *psum;
    long double swap_y, rtol, ytry, ysave, TINY = 1.0e-10;
    long double *y;
    Vector param_passed;
    long double tol = fx_param_double(opt_module_,"tolerance", 1.0e-7);
    index_t NMAX = fx_param_int(opt_module_, "MAX_FUNC_EVAL", 50000);

    param_passed.Init(dim);
    psum = (double*)malloc(dim * sizeof(double));
    num_func_eval = 0;
    y = (long double*)malloc(mpts*sizeof(long double));
    for(i = 0; i < mpts; i++) {	    
      param_passed.CopyValues(pts[i]);
      y[i] = (*func_ptr_)(param_passed,data());      
      
    }
    
  
    for(;;) {
      ilo = 0;
      ihi = y[0] > y[1] ? (inhi = 1,0) : (inhi = 0,1);
      for( i = 0; i < mpts; i++ ) {
	if(y[i] <= y[ilo]) ilo = i;
	if(y[i] > y[ihi]) {
	  inhi = ihi;
	  ihi = i;
	}
	else if((y[i] > y[inhi])&&(i != ihi)) inhi = i;
      }
		
      rtol = 2.0 * fabs(y[ihi] - y[ilo]) / ( fabs(y[ihi]) + fabs(y[ilo]) + TINY ) ;
      if(rtol < tol) {
	swap_y = y[0];
	y[0] = y[ilo];
	y[ilo] = swap_y;
	for( i = 0; i < dim; i++ ) {
	  swap = pts[0][i];
	  pts[0][i] =  pts[ilo][i] ;
	  pts[ilo][i] = swap;
	}
	fx_format_result(opt_module_,"min_obtained","%Lf", y[0]);
	break;
      }
      if(num_func_eval > NMAX){
	fx_format_result(opt_module_,"min_obtained","%Lf", y[ilo]);
	NOTIFY("Maximum number of function evaluations exceeded");
	break;
      }
      num_func_eval += 2;
		
      // Beginning a new iteration. 
      // Extrapolating by a factor of -1.0 through the face of the simplex
      // across from the high point, i.e, reflect the simplex from the high point
      for( j = 0 ; j < dim ; j++ ){
	sum = 0.0;
	for( i = 0 ; i < mpts ; i++ ) 
	  if (i != ihi)
	    sum += pts[i][j];
      
	psum[j] = sum / dim;
      }

      ytry = ModSimplex_(pts, y, psum, ihi, -1.0);
      if( ytry <= y[ilo] ) {	
	// result better than best point 
	// so additional extrapolation by a factor of 2
	ytry = ModSimplex_(pts, y, psum, ihi, 2.0);
      }
      else if( ytry >= y[ihi] ) { 
	// result worse than the worst point 
	// so there is a lower intermediate point, 
	// i.e., do a one dimensional contraction
	ysave = y[ihi];

	ytry = ModSimplex_(pts, y, psum, ihi, 0.5);
	if( ytry > y[ihi] ) { 
	  // Can't get rid of the high point, 
	  // try to contract around the best point
	  for( i = 0; i < mpts; i++ ) {
	    if( i != ilo ) {
	      for( j = 0; j < dim; j++ ) {
		pts[i][j] = psum[j] = 0.5 * ( pts[i][j] + pts[ilo][j] );
	      }
	      param_passed.CopyValues(psum);
	      y[i] = (*func_ptr_)(param_passed, data());	      
	    }
	  }
	  num_func_eval += dim;
	  for( j = 0 ; j < dim ; j++ ){
	    sum = 0.0;
	    for( i = 0 ; i < mpts ; i++ )
	      if (i != ihi)
		sum += pts[i][j];
	    psum[j] = sum / dim;
	  }
	}
      }
      else --num_func_eval;
    }
    fx_format_result(opt_module_, "func_evals", "%d", num_func_eval);
    return;
  }

  long double ModSimplex_(double **pts, long double *y, double *psum,
			  index_t ihi, float fac) {

 
    index_t j, dim = dimension();
    long double ytry;
    double *ptry;
    Vector param_passed;
	
    param_passed.Init(dim);
    ptry = (double*) malloc (dim * sizeof(double));
    for (j = 0; j < dim; j++) {
      ptry[j] = psum[j] * (1 - fac) + pts[ihi][j] * fac;
    }
    param_passed.CopyValues(ptry);
    ytry = (*func_ptr_)(param_passed, data());
    
    if (ytry < y[ihi]) {
      y[ihi] = ytry;
      for (j = 0; j < dim; j++) {
	pts[ihi][j] = ptry[j];
      }
    }
    return ytry;
  }

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

  void Eval(double *pt){

    index_t n = dimension(), iters;
    index_t i, its, MAXIMUM_ITERATIONS = fx_param_int(opt_module_,"MAX_ITERS",500);
    long double temp_1, temp_2, temp_3, temp_4, f_previous, f_min, 
      maximum_step_length, sum = 0.0, sumdg, sumxi, temp, test;
    Vector dgrad, grad, hdgrad, xi;
    Vector pold, pnew;
    Matrix hessian;
    double EPSILON = fx_param_double(opt_module_, "EPSILON", 3.0e-8);
    fx_format_param(opt_module_, "TOLERANCE", "%lf", 1.0e-5);
    double TOLERANCE = fx_param_double_req(opt_module_, "TOLERANCE");
    double MAX_STEP_SIZE = fx_param_double(opt_module_, "MAX_STEP_SIZE", 100.0);
    double g_tol = fx_param_double(opt_module_, "gtol", 1.0e-7);

    dgrad.Init(n);
    grad.Init(n);
    hdgrad.Init(n);
    hessian.Init(n,n);
    pnew.Init(n);
    xi.Init(n);
    pold.Copy(pt,n);
    f_previous = (*func_ptr_)(pold, data(), &grad);
    Vector tmp;
    tmp.Init(n);
    tmp.SetAll(1.0);
    hessian.SetDiagonal(tmp);
    la::ScaleOverwrite(-1.0, grad, &xi);
  
    sum = la::Dot(pold, pold);
    double fmax;
    if( sqrt(sum) > (float)n ) {
      fmax = sqrt(sum);
    }
    else { 
      fmax = (float)n;
    }
    maximum_step_length = MAX_STEP_SIZE*fmax;

    for(its = 0; its < MAXIMUM_ITERATIONS; its++) {
      
      dgrad.CopyValues(grad);
      LineSearch_(pold, f_previous, &grad, &xi,
		  &pnew, &f_min, maximum_step_length);
      f_previous = f_min;
      la::SubOverwrite(pold, pnew, &xi);
      pold.CopyValues(pnew);

      for(i = 0; i < n; i++) {
	pt[i] = pold.get(i);
      }

      test = 0.0;
      for(i = 0; i < n; i++){
	if(fabs(pold.get(i)) > 1.0) fmax = fabs(pold.get(i));
	else{ fmax = 1.0; }
	temp = fabs(xi.get(i)) / fmax;
	if(temp > test) test = temp;
      }
      if(test < TOLERANCE) {
	iters = its;
	fx_format_result(opt_module_, "iters", "%d", iters);
	fx_format_result(opt_module_,"min_obtained","%Lf", f_previous);
	return;
      }

      test = 0.0;
      if(f_min > 1.0) temp_1 = f_min;
      else{ temp_1 = 1.0; }

      for(i = 0; i < n; i++) {
	if(fabs(pold.get(i)) > 1.0) fmax = pold.get(i);
	else{ fmax = 1.0; }

	temp = fabs(grad.get(i))*fmax / temp_1;
	if(temp > test) test = temp;
      }
      if(test < g_tol) {
	iters = its;
	fx_format_result(opt_module_, "iters", "%d", iters);
	fx_format_result(opt_module_,"min_obtained","%Lf", f_previous);
	return;
      }

      la::SubFrom(grad, &dgrad);
      la::Scale(-1.0, &dgrad);
      la::MulOverwrite(hessian,dgrad, &hdgrad);

      temp_2 = la::Dot(dgrad, xi);
      temp_4 = la::Dot(dgrad, hdgrad);
      sumdg = la::Dot(dgrad, dgrad);
      sumxi = la::Dot(xi, xi);

      if (temp_2 > sqrt(EPSILON*sumdg*sumxi)) {
	temp_2 = 1.0 / temp_2;
	temp_3 = 1.0 / temp_4;

	la::ScaleOverwrite(temp_2, xi, &dgrad);
	la::AddExpert((-1.0*temp_3), hdgrad, &dgrad);

	Matrix co, ro, tmp;
	co.AliasColVector(xi);
	ro.AliasRowVector(xi);
	la::MulInit(co, ro, &tmp);
	la::AddExpert(temp_2, tmp, &hessian);

	co.Destruct();
	ro.Destruct();
	tmp.Destruct();
	co.AliasColVector(hdgrad);
	ro.AliasRowVector(hdgrad);
	la::MulInit(co, ro, &tmp);
	la::AddExpert((-1.0*temp_3), tmp, &hessian);

	co.Destruct();
	ro.Destruct();
	tmp.Destruct();
	co.AliasColVector(dgrad);
	ro.AliasRowVector(dgrad);
	la::MulInit(co, ro, &tmp);
	la::AddExpert(temp_4, tmp, &hessian);
      }
      la::MulOverwrite(hessian, grad, &xi);
      la::Scale((-1.0), &xi);
    }
    NOTIFY("Too many iterations in Quasi Newton\n");
    fx_format_result(opt_module_,"min_obtained","%Lf", f_previous);
  }

  void LineSearch_(Vector pold, long double fold, Vector *grad,
		   Vector *xi, Vector *pnew, long double *f_min,
		   long double maximum_step_length){

    index_t i, n = dimension();
    long double a, step_length, previous_step_length = 0.0, 
      minimum_step_length, b, disc, previous_f_value = 0.0,
      rhs1, rhs2, slope, sum, temp, test, temp_step_length,
      MIN_DECREASE = 1.0e-4, TOLERANCE = 1.0e-7;
    
    sum = la::Dot(*xi, *xi);
    sum = sqrt(sum);
    if(sum > maximum_step_length) {
      la::Scale((maximum_step_length/sum), xi);
    }
    slope = la::Dot(*grad, *xi);
    if(slope >= 0.0){
      return;
    }
    test = 0.0;
    for(i = 0; i < n; i++) {
      double fmax;
      fmax = (fabs(pold.get(i)) > 1.0 ? fabs(pold.get(i)) : 1.0);
      temp = fabs((*xi).get(i)) / fmax;
      if(temp > test) test = temp;
    }

    minimum_step_length = TOLERANCE/test;
    step_length = 1.0;
    for(;;) {

      pnew->CopyValues(pold);
      la::AddExpert(step_length, *xi, pnew);

      *f_min = (*func_ptr_)((*pnew), data(), grad);
      if(step_length < minimum_step_length) {
	pnew->CopyValues(pold);
	return;
      }
      else if( *f_min <= fold + MIN_DECREASE*step_length*slope) {
	return;
      }
      else {
	if (step_length == 1.0) {
	  temp_step_length = -slope/(2.0*(*f_min - fold - slope));
	}
	else {
	  rhs1 = *f_min - fold - step_length*slope;
	  rhs2 = previous_f_value - fold - previous_step_length*slope;
	  a = (rhs1 / (step_length*step_length) 
	       - rhs2/(previous_step_length*previous_step_length))
	    / (step_length-previous_step_length);
	  b = (-previous_step_length*rhs1/(step_length*step_length)
	       +step_length*rhs2/(previous_step_length*previous_step_length)) 
	    / (step_length - previous_step_length);
	  if(a == 0.0) {
	    temp_step_length = -slope / (2.0*b);
	  }
	  else {
	    disc = b*b - 3.0*a*slope;
	    if(disc < 0.0) {
	      temp_step_length = 0.5*step_length;
	    }
	    else if (b <= 0.0) {
	      temp_step_length = (-b+sqrt(disc))/(3.0*a);
	    }
	    else {
	      temp_step_length = -slope / (b+sqrt(disc));
	    }
	  }
	  if(temp_step_length > 0.5*step_length) {
	    temp_step_length = 0.5*step_length;
	  }
	}
      }
      previous_step_length = step_length;
      previous_f_value = *f_min;
      step_length = (temp_step_length > 0.1*step_length 
		     ? temp_step_length : 0.1*step_length);
    }
  }
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

  void Eval(double *pt){

    index_t iters;
    index_t MAXIMUM_ITERATIONS = fx_param_int(opt_module_,"MAX_ITERS",100);
    double EPSILON = fx_param_double(opt_module_, "EPSILON", 1.0e-5);
    fx_format_param(opt_module_, "TOLERANCE", "%lf", 0.001);
    double TOLERANCE = fx_param_double_req(opt_module_, "TOLERANCE");
    // double MAX_STEP_SIZE = fx_param_double(opt_module_, 
    //					   "MAX_STEP_SIZE", 100.0);
    index_t dim = fx_param_int_req(opt_module_, "param_space_dim");
    Vector pold, pnew, grad;
    long double f_old, f_new;
    double scale, alpha = 0.1, gamma;
    long double p_tol = 0.0, f_tol = 0.0;

    // have to decide what to assign alpha value as 
    // step lengths are crucial because this is 
    // ending up oscillating close to the optimal
    // hence never actually reaching the optimal

    pold.Init(dim);
    pnew.Init(dim);
    pold.CopyValues(pt);
    grad.Init(dim);

    f_old = (*func_ptr_)(pold, data(), &grad);
    printf("first val: %Lf\n", f_old);
 
    // Here we are doing the normal gradient step
    // scale = || - \nabla_\theta f(X, \theta_k) ||
    // \theta_{k+1} = \theta_k - 
    //                alpha * \nabla_\theta f(X,\theta_k) / scale;

    for (iters = 0; iters < MAXIMUM_ITERATIONS; iters++) {

      scale = sqrt(la::Dot(grad, grad));
      gamma = - alpha / scale;
      pnew.SetZero();
      la::AddTo(pold, &pnew);
      la::AddExpert(gamma, grad, &pnew);

      Vector diff;
      la::SubInit(pnew, pold, &diff);
      p_tol = sqrt(la::Dot(diff, diff));

      f_new = (*func_ptr_)(pnew, data(), &grad);
      f_tol = fabs(f_new - f_old);

      if (((f_tol < EPSILON) && (p_tol < TOLERANCE)) || (scale < EPSILON)) {
	fx_format_result(opt_module_, "iters", "%d", iters+1);
	fx_format_result(opt_module_,"min_obtained","%Lf", f_old);
	for (index_t i = 0; i < dim; i++) {
	  pt[i] = pold.get(i);
	}
	return;
      }

      pold.CopyValues(pnew);
      f_old = f_new;
    }

    NOTIFY("Too many iterations in Gradient Descent\n");
    fx_format_result(opt_module_,"min_obtained","%Lf", f_old);
    for(index_t i = 0; i < dim; i++) {
      printf("%lf, ", pold.get(i));
    }
    printf("\nfinal val: %Lf\n p_tol : %Lf, f_tol : %Lf, iters : %"LI"d\n", f_old, p_tol, f_tol, iters);
    return;
  }
  
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

  void Eval(double *pt){

    index_t iters;
    index_t MAXIMUM_ITERATIONS = fx_param_int(opt_module_,"MAX_ITERS",100);
    double EPSILON = fx_param_double(opt_module_, "EPSILON", 1.0e-5);
    fx_format_param(opt_module_, "TOLERANCE", "%lf", 0.001);
    double TOLERANCE = fx_param_double_req(opt_module_, "TOLERANCE");
    // double MAX_STEP_SIZE = fx_param_double(opt_module_, 
    //					   "MAX_STEP_SIZE", 100.0);
    index_t dim = fx_param_int_req(opt_module_, "param_space_dim");
    index_t num_batch = fx_param_int(opt_module_, "BATCHES",50);
    Vector pold, pnew, grad;
    long double f_old, f_new;
    double scale, alpha = 0.1, gamma;
    long double p_tol = 0.0, f_tol = 0.0;
    Matrix data_batched;
    index_t batch_size = data().n_cols() / num_batch;

    // have to decide what to assign alpha value as 
    // step lengths are crucial because this is 
    // ending up oscillating close to the optimal
    // hence never actually reaching the optimal

    pold.Init(dim);
    pnew.Init(dim);
    pold.CopyValues(pt);
    grad.Init(dim);
    data_batched.Copy(data());

    f_old = (*func_ptr_)(pold, data(), &grad);
    printf("first val: %Lf\n", f_old);
 
    // Here we are doing the normal gradient step
    // scale = || - \nabla_\theta f(X_t, \theta_t) ||
    // \theta_{t+1} = \theta_t - 
    //                alpha * \nabla_\theta f(X_t,\theta_t) / scale;

    for (iters = 0; iters < MAXIMUM_ITERATIONS; iters++) {

      // Now going through the data batchwise
      for (index_t in = 0; in < num_batch; in++) {

	scale = sqrt(la::Dot(grad, grad));
	gamma = - alpha / scale;
	pnew.SetZero();
	la::AddTo(pold, &pnew);
	la::AddExpert(gamma, grad, &pnew);

	Vector diff;
	la::SubInit(pnew, pold, &diff);
	p_tol = sqrt(la::Dot(diff, diff));

	// using a batch
	Matrix single_batch;
	index_t st_pt = in * batch_size;
	data_batched.MakeColumnSlice(st_pt, batch_size, &single_batch);
	f_new = (*func_ptr_)(pnew, single_batch, &grad);
	f_tol = fabs(f_new - f_old);

	if ((f_tol < EPSILON) && (p_tol < TOLERANCE)) {
	  fx_format_result(opt_module_, "iters", "%d", iters+1);
	  fx_format_result(opt_module_,"min_obtained","%Lf", f_old);
	  for (index_t i = 0; i < dim; i++) {
	    pt[i] = pold.get(i);
	  }
	  printf("iters: %"LI"d, min: %Lf\n", iters, f_old);
	  return;
	}

	pold.CopyValues(pnew);
	f_old = f_new;
 
      }

      // permuting the data matrix
      data_batched.Destruct();
      PermuteMatrix_(data(), &data_batched);
      //printf("data permuted\n");
    }

    NOTIFY("Too many iterations in Stochastic Gradient Descent\n");
    fx_format_result(opt_module_,"min_obtained","%Lf", f_old);
    for(index_t i = 0; i < dim; i++) {
      printf("%lf, ", pold.get(i));
    }
    long double f_final = (*func_ptr_)(pold, data(), &grad);
    printf("\nfinal val: %Lf\n p_tol : %Lf, f_tol : %Lf, iters : %"LI"d\n",
	   f_final, p_tol, f_tol, iters);
    return;
  }

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

class SMD_SingleStep {

 private:
  index_t dimension_;
  Matrix data_;
  long double (*func_ptr_)(Vector&, const Matrix&, Vector*);
  datanode *opt_module_;

 public:

  SMD_SingleStep(){
  }

  ~SMD_SingleStep(){ 
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

  void Eval(double *pt){

    index_t iters;
    index_t MAXIMUM_ITERATIONS = fx_param_int(opt_module_,"MAX_ITERS",100);
    double EPSILON = fx_param_double(opt_module_, "EPSILON", 1.0e-2);
    fx_format_param(opt_module_, "TOLERANCE", "%lf", 0.01);
    double TOLERANCE = fx_param_double_req(opt_module_, "TOLERANCE");
    index_t dim = fx_param_int_req(opt_module_, "param_space_dim");
    index_t num_batch = fx_param_int(opt_module_, "BATCHES",50);
    Vector pold, pnew, grad, prev_grad;
    long double f_old, f_new;
    double scale, scale_prev, eta = 0.01, gamma, mu = 0.1;
    double p_tol = 0.0, f_tol = 0.0;
    Matrix data_batched;
    index_t batch_size = data().n_cols() / num_batch;

    // have to decide how to chose starting value
    // of alpha (right now it is just 1). 
    // also have to decide the value for the 
    // meta parameter mu (right now it is 
    // arbitrarily chosen as 0.1)

    pold.Init(dim);
    pnew.Init(dim);
    pold.CopyValues(pt);
    grad.Init(dim);
    prev_grad.Init(dim);
    data_batched.Copy(data());

    f_old = (*func_ptr_)(pold, data(), &grad);
    printf("first val: %Lf\n", f_old);
    scale = sqrt(la::Dot(grad, grad));

    // Here we are doing the gradient step
    // scale = || - \nabla_\theta f(X_t, \theta_t) ||
    // \theta_{t+1} = \theta_t - 
    //                \eta_t * \nabla_\theta f(X_t,\theta_t) / scale;

    for (iters = 0; iters < MAXIMUM_ITERATIONS; iters++) {

      // Now going through the data batchwise
      for (index_t in = 0; in < num_batch; in++) {

	// instead of scaling the gradient, how about using low values
	// of the step sizes, because scaling the gradients result
	// in the gradient being significant even when it is close
	// to the optimal
	//	gamma = - eta / scale;
	gamma = -eta;
	pnew.SetZero();
	la::AddTo(pold, &pnew);
	la::AddExpert(gamma, grad, &pnew);


	// using a batch
	Matrix single_batch;
	index_t st_pt = in * batch_size;
	data_batched.MakeColumnSlice(st_pt, batch_size, &single_batch);
	prev_grad.CopyValues(grad);
	f_new = (*func_ptr_)(pnew, single_batch, &grad);

	// Terminating conditions
	// |f_t+1 - f_t| < epsilon & ||\theta_t+1 - \theta_t|| < delta
	Vector diff;
	la::SubInit(pnew, pold, &diff);
	f_tol = fabs(f_new - f_old);
	p_tol = sqrt(la::Dot(diff, diff));

	// but instead if we used the condition
	// ||grad_t|| < epsilon' & ||\theta_t+1 - \theta_t|| < delta
	// if ((f_tol < EPSILON) && (p_tol < TOLERANCE)) {

	// this doesn't work either, same problem
	// if ((scale < EPSILON) && (p_tol < TOLERANCE)){

	// using just the point in the param_space 
	// which refuses to move
	if (p_tol < TOLERANCE) { 
	  // rejected because stops too early, need the check
	  // the overall gradient is small
	  Vector temp_grad;
	  temp_grad.Init(dim);
	  long double f_final = (*func_ptr_)(pold, data(), &temp_grad);
	  double temp_grad_val = sqrt(la::Dot(temp_grad, temp_grad));
	  if (temp_grad_val < EPSILON) {

	    fx_format_result(opt_module_, "iters", "%d", iters+1);
	    fx_format_result(opt_module_,"min_obtained","%Lf", f_final);
	    for (index_t i = 0; i < dim; i++) {
	      pt[i] = pold.get(i);
	    }
	    printf("iters: %"LI"d\n", iters); 
	    for(index_t i = 0; i < dim; i++) {
	      printf("%lf, ", pold.get(i));
	    }
	    printf("\nfinal val: %Lf\n p_tol : %lf, iters : %"LI"d, g_tol : %lf\n", 
		   f_final, p_tol, iters, temp_grad_val);
	    
	    return;
	  }
	}

	pold.CopyValues(pnew);
	f_old = f_new;

	// updating the step size as per the following
	// \eta_i = \eta_{i-1} * max(0.5, 1 + mu * \eta_{i-1} * 
	//                           \nabla_\theta f_{i-1}'* 
	//                           \nabla_\theta f_i 
	//                          )
	scale_prev = scale;
	scale = sqrt(la::Dot(grad, grad));
	//	double temp_eta = 1 + mu * eta * (la::Dot(grad, prev_grad)) / 
	//(scale * scale_prev);
	double temp_eta = 1 + mu * eta * (la::Dot(grad, prev_grad));
	eta = eta * ((0.5 > temp_eta)?0.5 : temp_eta);
      }

      // permuting the data matrix
      data_batched.Destruct();
      PermuteMatrix_(data(), &data_batched);
      //printf("data permuted\n");
    }

    NOTIFY("Too many iterations in Stochastic Meta Descent\n");
    fx_format_result(opt_module_,"min_obtained","%Lf", f_old);
    for(index_t i = 0; i < dim; i++) {
      printf("%lf, ", pold.get(i));
    }
    long double f_final = (*func_ptr_)(pold, data(), &grad);
    scale = sqrt(la::Dot(grad, grad));
    printf("\nfinal val: %Lf, p_tol : %lf, iters : %"LI"d, g_tol : %lf\n",
	   f_final, p_tol, iters, scale);
    return;
  }

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

#endif
