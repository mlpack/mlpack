#include <fastlib/fastlib.h>
#include "optimizers_reloaded.h"


void QuasiNewton::Eval(double *pt){

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

void QuasiNewton::LineSearch_(Vector pold, long double fold, Vector *grad,
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



void GradientDescent::Eval(double *pt){

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


void SGD::Eval(double *pt){

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



void SMD::Eval(double *pt){

  index_t iters;
  index_t MAXIMUM_ITERATIONS = fx_param_int(opt_module_,"MAX_ITERS",100);
  // double EPSILON = fx_param_double(opt_module_, "EPSILON", 1.0e-2);
  double TOLERANCE = fx_param_double(opt_module_, "TOLERANCE", 1.0e-2);
  index_t dim = fx_param_int_req(opt_module_, "param_space_dim");
  index_t num_batch = fx_param_int(opt_module_, "BATCHES",50);
  Vector pold, pnew, eta, grad, v_vec, one_vec;
  long double f;//f_new;
  //double scale, scale_prev, gamma;
  double mu = fx_param_double(opt_module_, "MU", 0.1);
  // double lambda = fx_param_double(opt_module_, "LAMBDA", 0.99);
  double p_tol = 0.0;//, f_tol = 0.0;
  Matrix data_batched, single_batch;
  index_t batch_size = fx_param_int(opt_module_,"BATCH_SIZE",
				    data().n_cols() / num_batch);

  //    fx_clear_param(opt_module_,"BATCHES");
  num_batch = fx_param_int(opt_module_,"BATCHES",
			   data().n_cols() / batch_size);

  one_vec.Init(dim);
  one_vec.SetAll(1.0);
  // to decide how to chose starting value
  // of alpha (right now it is just 1). 
  // also have to decide the value for the 
  // meta parameter mu (right now it is 
  // arbitrarily chosen as 0.1)
  
  // p_0
  pold.Copy(pt, dim);
  //pnew.Init(dim);
  grad.Init(dim);
  //prev_grad.Init(dim);
  eta.Init(dim);
  // eta_0
  eta.SetAll(0.1);
  
  PermuteMatrix_(data(), &data_batched);
  // x_0
  data_batched.MakeColumnSlice(0, batch_size, &single_batch);
  // [f_0, g_0] = f(p_0, x_0)
  f = (*func_ptr_)(pold, data(), &grad);
  // v_1 = -g_0 X eta_0
  HadamardInit(grad, eta, &v_vec);
  la::Scale(-1.0, &v_vec);
  // p_1 = p_0 - g_0 X eta_0 
  //     = p_0 + v_1
  la::AddInit(pold, v_vec, &pnew);

  data_batched.Destruct();
  PermuteMatrix_(data(), &data_batched);

  for (iters = 0; iters < MAXIMUM_ITERATIONS; iters++) {

    // Now going through the data batchwise
    for (index_t in = 0; in < num_batch; in++) {

      // Terminating conditions
      // ||p_t+1 - p_t|| < TOL

      Vector diff;
      la::SubInit(pnew, pold, &diff);
      p_tol = sqrt(la::Dot(diff, diff));

      // using just the point in the param_space 
      // which refuses to move
      if (p_tol < TOLERANCE) { 
	// rejected because stops too early, need the check
	// the overall gradient is small
	Vector temp_grad;
	temp_grad.Init(dim);
	long double f_final = (*func_ptr_)(pnew, data(), &temp_grad);
	// but maybe we can skip that now
	// double temp_grad_val = sqrt(la::Dot(temp_grad, temp_grad));
	// if (temp_grad_val < EPSILON) {

	fx_format_result(opt_module_, "iters", "%d", iters+1);
	fx_format_result(opt_module_,"min_obtained","%Lf", f_final);
	for (index_t i = 0; i < dim; i++) {
	  pt[i] = pold.get(i);
	}
	// printf("iters: %"LI"d\n", iters); 
	// for(index_t i = 0; i < dim; i++) {
	//   printf("%lf, ", pold.get(i));
	// }
	printf("\nfinal val: %Lf\n", f_final);
	printf("p_tol : %lf, iters : %"LI"d, batch_number : %"LI"d\n", 
	       p_tol, iters, in);
	    
	return;
	// }
      }

      // x_t+1
      index_t st_pt = in * batch_size;
      single_batch.Destruct();
      data_batched.MakeColumnSlice(st_pt, batch_size, &single_batch);
      
      // [f_t+1, g_t+1] = f(p_t+1, x_t+1)
      f = (*func_ptr_)(pnew, single_batch, &grad);

      // [f_t+1 g_t+1 H_t+1*v_t+1] = f(p_t+1, x_t+1, v_t+1)
      // Vector hess_v;
      // f = (*func_ptr)(pnew, single_batch, &grad, v_vec, &hess_v);
	
      // But since the L2 function is pretty awesome, we need 
      // to make the calling a little diff
      // f_new = (*func_ptr_stoc_)(pnew, single_batch, &grad, num_batch);

      // eta_t+1 = eta_t * max(0.5, 1 + mu * g_t+1 X v_t+1)

      Vector one_mu_grad_v, max_half_one_mu_grad_v;
      HadamardInit(grad, v_vec, &one_mu_grad_v);
      la::Scale(mu, &one_mu_grad_v);
      la::AddTo(one_vec, &one_mu_grad_v);

      max_half_one_mu_grad_v.Copy(one_mu_grad_v);
      for (index_t j = 0; j < max_half_one_mu_grad_v.length(); j++) {
	if (max_half_one_mu_grad_v.get(j) < 0.5) {
	  max_half_one_mu_grad_v.ptr()[j] = 0.5;
	}
      }

      HadamardTo(max_half_one_mu_grad_v, &eta);

      // Current:
      // v_t+2 = eta_t+1 X g_t+1
      HadamardOverwrite(eta, grad, &v_vec);

      // Later:
      // v_t+2 = lambda * v_t+1 + eta_t+1 X (g_t+1 - lambda * H_t+1 * v_t+1)
      //       Current: H_t+1 = g_t+1 * g_t+1'
 
      //       Matrix grad_mat, grad_mat_trans, hess;
      //       grad_mat.AliasColVector(grad);
      //       grad_mat_trans.AliasRowVector(grad);
      //       la::MulInit(grad_mat, grad_mat_trans, &hess);

      //       Vector lambda_hess_v, eta_prod;
      //       la::MulInit(hess, v_vec, &lambda_hess_v);
      //       la::Scale(lambda, &lambda_hess_v);
      //       la::SubInit(lambda_hess_v, grad, &eta_prod);
      //       HadamardTo(eta, &eta_prod);

      //       la::Scale(lambda, &v_vec);
      //       la::AddTo(eta_prod, &v_vec);

      //       Later : Calculate H_t+1 * v_t+1 directly

      //       Vector lambda_hess_v, eta_prod;
      //       la::ScaleInit(lambda, hess_v, &lambda_hess_v);
      //       la::SubInit(lambda_hess_v, grad, &eta_prod);
      //       HadamardTo(eta, &eta_prod);

      //       la::Scale(lambda, &v_vec);
      //       la::AddTo(eta_prod, &v_vec);

      // p_t+2 = p_t+1 - eta_t+1 X g_t+1
      pold.CopyValues(pnew);
      Vector eta_grad;
      HadamardInit(eta, grad, &eta_grad);
      la::SubFrom(eta_grad, &pnew);
    }

    // permuting the data matrix
    data_batched.Destruct();
    PermuteMatrix_(data(), &data_batched);
  }
  fflush(NULL);
  NOTIFY("Too many iterations in Stochastic Meta Descent");
  pt = pold.ptr();
  //  for(index_t i = 0; i < dim; i++) {
  //     pt[i] = pold.get(i);
  //   }
  long double f_final = (*func_ptr_)(pold, data(), &grad);
  // scale = sqrt(la::Dot(grad, grad));
  printf("\nfinal val: %Lf, p_tol : %lf, iters : %"LI"d\n",
	 f_final, p_tol, iters);
  fx_format_result(opt_module_,"min_obtained","%Lf", f_final);
   
  return;
}
