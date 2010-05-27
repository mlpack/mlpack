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
 * @file optimizers.cc
 *
 * Implementation of the optimizers
 *
 */

#include "optimizers.h"

void NelderMead::Eval(double **pts) {

  index_t dim = dimension(), num_func_eval;
  index_t i, j, ihi, ilo, inhi,mpts = dim + 1;
  double sum, swap, *psum;
  long double swap_y, rtol, ytry, ysave, TINY = 1.0e-10;
  long double *y;
  Vector param_passed;
  long double tol = fx_param_double(opt_module_,"tolerance", 1.0e-5);
  index_t NMAX = fx_param_int(opt_module_, "MAX_FUNC_EVAL", 50000);

  param_passed.Init(dim);
  psum = (double*)malloc(dim * sizeof(double));
  num_func_eval = 0;
  y = (long double*)malloc(mpts*sizeof(long double));
  for(i = 0; i < mpts; i++) {
    param_passed.CopyValues(pts[i]);
    y[i] = (*func_ptr_)(param_passed, data());
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
      break;
    }
    if(num_func_eval > NMAX){
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
  fx_result_int(opt_module_, "func_evals", num_func_eval);
  return;
}

long double NelderMead::ModSimplex_(double **pts, long double *y,
				    double *psum, index_t ihi,
				    float fac) {

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

void QuasiNewton::Eval(double *pt) {

  index_t n = dimension(), iters;
  index_t i, its, MAXIMUM_ITERATIONS = fx_param_int(opt_module_,"MAX_ITERS",200);
  long double temp_1, temp_2, temp_3, temp_4, f_previous, f_min, 
    maximum_step_length, sum = 0.0, sumdg, sumxi, temp, test;
  Vector dgrad, grad, hdgrad, xi;
  Vector pold, pnew;
  Matrix hessian;
  double EPSILON = fx_param_double(opt_module_, "EPSILON", 3.0e-8);
  double TOLERANCE = fx_param_double(opt_module_, "TOLERANCE", 1.0e-5);
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
      fx_result_int(opt_module_, "iters", iters);
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
      fx_result_int(opt_module_, "iters", iters);
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
}

void QuasiNewton::LineSearch_(Vector pold, long double fold,
			      Vector *grad, Vector *xi,
			      Vector *pnew, long double *f_min,
			      long double maximum_step_length) {

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
