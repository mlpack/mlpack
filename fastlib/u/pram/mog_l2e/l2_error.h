/**
  * Description of the L2 error function we are trying to minimize
  *
  *
  */

#include "fastlib/fastlib.h"
#include "mog.h"
//#include "phi.h"

long double l2_error(Matrix&, index_t, double*);

long double l2_error(Matrix&, index_t, double*, Vector*);

long double calc_reg(MoG&);

long double calc_reg(MoG&, Vector*);

long double calc_fit(Matrix&, MoG&);

long double calc_fit(Matrix&, MoG&, Vector*);

long double mod_simplex(double**, long double[], double[], index_t,
			long double (*funk)(Matrix&, index_t, double*),
			index_t, float, Matrix&, index_t) ;

bool polytope(double**, long double*, index_t ndim, long double ftol,
	      long double (*funk)(Matrix&, index_t, double*),index_t*, 
	      Matrix& data, index_t k_comp);

void line_search(index_t, Vector, long double, Vector*, Vector*, Vector*,
		 long double*, long double, index_t*, 
		 long double (*funk)(Matrix&, index_t, double*, Vector*), 
		 Matrix&, index_t);

void quasi_newton(double*, index_t, double, index_t*, long double*, 
		  long double (*func)(Matrix&, index_t, double*, Vector*), 
		  Matrix&, index_t);

long double l2_error(Matrix& d, index_t num_gauss, double* theta) {
  MoG mog;
  long double reg, fit, l2e; 
  index_t dim, number_of_points;
	
  number_of_points = d.n_cols();
  dim = d.n_rows();
  mog.MakeModel(num_gauss, dim, theta);
  reg = calc_reg(mog);
  fit = calc_fit(d, mog);
  l2e = reg - 2 * fit / number_of_points ;
  return l2e;
}
	

long double l2_error(Matrix& d, index_t num_gauss, double* theta, Vector *g_l2_error) {
  MoG mog;
  long double reg, fit, l2e; 
  index_t dim, number_of_points;
  Vector g_reg,g_fit,tmp_fit;
	
  number_of_points = d.n_cols();
  dim = d.n_rows();
  mog.MakeModelWithGradients(num_gauss, dim, theta);
  reg = calc_reg(mog, &g_reg);
  fit = calc_fit(d,mog, &g_fit);
  l2e = reg - 2*fit / number_of_points ;
  double alpha = -2.0 / number_of_points;
  la::ScaleInit(alpha, g_fit, &tmp_fit);
  la::SubOverwrite(tmp_fit, g_reg, g_l2_error);
  return l2e;
}


long double calc_reg(MoG& mog) {
  Matrix phi_mu, sum_covar;
  Vector x, y;
  long double reg, tmpVal;
  index_t num_gauss, dim;
	
  num_gauss = mog.number_of_gaussians();
  dim = mog.dimension();

  phi_mu.Init(num_gauss, num_gauss);
  sum_covar.Init(dim, dim);
  x.Copy(mog.omega());
	
  for(index_t k = 1; k < num_gauss; k++) {
    for(index_t j = 0; j < k; j++) {
      la::AddOverwrite(mog.sigma(k), mog.sigma(j), &sum_covar);
      tmpVal = phi(mog.mu(k), mog.mu(j), sum_covar);
      phi_mu.set(j, k, tmpVal);
      phi_mu.set(k, j, tmpVal);
    }			
  }
  for(index_t k = 0; k < num_gauss; k++) {
    la::ScaleOverwrite(2, mog.sigma(k), &sum_covar);
    tmpVal = phi(mog.mu(k), mog.mu(k), sum_covar);
    phi_mu.set(k, k, tmpVal);
  }
  la::MulInit(x, phi_mu, &y);
  reg = la::Dot(x, y);
	
  return reg;
}


long double calc_reg(MoG& mog, Vector *g_reg){
  Matrix phi_mu, sum_covar;
  Vector x, y;
  long double reg, tmpVal;
  index_t num_gauss, dim;
  Vector df_dw, g_omega;
  ArrayList<Vector> g_mu, g_sigma;
  ArrayList<ArrayList<Vector> > dp_d_mu, dp_d_sigma;

  num_gauss = mog.number_of_gaussians();
  dim = mog.dimension();

  phi_mu.Init( num_gauss, num_gauss );
  sum_covar.Init(dim, dim);
  x.Copy(mog.omega());
  g_mu.Init(num_gauss);
  g_sigma.Init(num_gauss);
  dp_d_mu.Init(num_gauss);
  dp_d_sigma.Init(num_gauss);
  for(index_t k = 0; k < num_gauss; k++){
    dp_d_mu[k].Init(num_gauss);
    dp_d_sigma[k].Init(num_gauss);
  }
  for(index_t k = 1; k < num_gauss; k++) {
    for(index_t j = 0; j < k; j++) {
      la::AddOverwrite(mog.sigma(k), mog.sigma(j), &sum_covar);
			
      ArrayList<Matrix> tmp_d_cov;
      Vector tmp_dp_d_sigma;
      
      tmp_d_cov.Init(dim*(dim+1));
      for(index_t i = 0; i < (dim*(dim + 1) / 2); i++){
	tmp_d_cov[i].Copy(mog.d_sigma(k)[i]);
	tmp_d_cov[(dim*(dim+1)/2)+i].Copy(mog.d_sigma(j)[i]);
      }
      
      tmpVal = phi(mog.mu(k),mog.mu(j),sum_covar,tmp_d_cov,&dp_d_mu[j][k],&tmp_dp_d_sigma);
      
      phi_mu.set(j, k, tmpVal);
      phi_mu.set(k, j, tmpVal);
			
      la::ScaleInit(-1.0, dp_d_mu[j][k], &dp_d_mu[k][j]);
			
      double *tmp_dp, *tmp_dp_1, *tmp_dp_2;
      tmp_dp = tmp_dp_d_sigma.ptr();
      tmp_dp_1 = (double*)malloc((tmp_dp_d_sigma.length()/2) * sizeof(double));
      tmp_dp_2 = (double*)malloc((tmp_dp_d_sigma.length()/2) * sizeof(double));
      for(index_t i = 0; i < (tmp_dp_d_sigma.length()/2); i++){
	tmp_dp_1[i] = tmp_dp[i];
	tmp_dp_2[i] = tmp_dp[(dim*(dim + 1) / 2) + i];
      }
      dp_d_sigma[j][k].Copy(tmp_dp_1, (dim*(dim + 1) / 2));
      dp_d_sigma[k][j].Copy(tmp_dp_2, (dim*(dim + 1) / 2));
    }
  }
	
  for(index_t k = 0; k < num_gauss; k++) {
    Vector junk;
    la::ScaleOverwrite(2, mog.sigma(k), &sum_covar);
    tmpVal = phi(mog.mu(k), mog.mu(k), sum_covar, mog.d_sigma(k), &junk, &dp_d_sigma[k][k]);
    phi_mu.set(k, k, tmpVal);
    dp_d_mu[k][k].Init(dim);
    dp_d_mu[k][k].SetZero();
  }
	
  // Calculating the reg value
  la::MulInit( x, phi_mu, &y );
  reg = la::Dot( x, y );
	
  // Calculating the g_omega values - a vector of size K-1
  la::ScaleInit(2.0,y,&df_dw);
  la::MulInit(mog.d_omega(),df_dw,&g_omega);
	
  // Calculating the g_mu values - K vectors of size D
  for(index_t k = 0; k < num_gauss; k++){
    g_mu[k].Init(dim);
    g_mu[k].SetZero();
    for(index_t j = 0; j < num_gauss; j++)
      la::AddExpert(x.get(j), dp_d_mu[j][k], &g_mu[k]);
    la::Scale((2.0 * x.get(k)), &g_mu[k]);
  }
	
  // Calculating the g_sigma values - K vectors of size D(D+1)/2
  for(index_t k = 0; k < num_gauss; k++){
    g_sigma[k].Init((dim*(dim + 1)) / 2);
    g_sigma[k].SetZero();
    for(index_t j = 0; j < num_gauss; j++)
      la::AddExpert(x.get(j), dp_d_sigma[j][k], &g_sigma[k]);
    la::Scale((2.0 * x.get(k)), &g_sigma[k]);
  }
	
  // Making the single gradient vector of size K*(D+1)*(D+2)/2
  double *tmp_g_reg;
  tmp_g_reg = (double*)malloc(((num_gauss*(dim + 1)*(dim + 2) / 2) - 1)*sizeof(double));
  index_t j = 0;
  for(index_t k = 0; k < g_omega.length(); k++)
    tmp_g_reg[k] = g_omega.get(k);
  j = g_omega.length();
  for(index_t k = 0; k < num_gauss; k++){
    for(index_t i = 0; i < dim; i++){
      tmp_g_reg[j + k*(dim) + i] = g_mu[k].get(i);
    }
    for(index_t i = 0; i < (dim*(dim+1)/2); i++){
      tmp_g_reg[j + num_gauss*dim + k*(dim*(dim+1) / 2) + i] = g_sigma[k].get(i);
    }
  }
  (*g_reg).Copy(tmp_g_reg, ((num_gauss*(dim+1)*(dim+2) / 2) - 1));
	
  return reg;
}


long double calc_fit(Matrix& data, MoG& mog) {
  long double fit;
  Matrix phi_x;
  Vector weights, x, y, identity_vector;
  index_t num_gauss, num_points;

  num_gauss = mog.number_of_gaussians();
  num_points = data.n_cols();
  phi_x.Init( num_gauss, num_points );
  weights.Copy(mog.omega()); 
  x.Init(data.n_rows());
  identity_vector.Init(num_points);
  identity_vector.SetAll(1);
	
  for(index_t k = 0; k < num_gauss; k++) {
    for(index_t i = 0; i < num_points; i++) {
      x.CopyValues(data.GetColumnPtr(i));
      phi_x.set(k, i, phi(x, mog.mu(k), mog.sigma(k)));
    }
  }

  la::MulInit(weights, phi_x, &y);
  fit = la::Dot(y, identity_vector); 
  return fit;
}


long double calc_fit(Matrix& data, MoG& mog, Vector *g_fit) {
  long double fit;
  Matrix phi_x;
  Vector weights, x, y, identity_vector;
  index_t num_gauss, num_points, dim;
  long double tmpVal;
  Vector g_omega,tmp_g_omega;
  ArrayList<Vector> g_mu, g_sigma;

  num_gauss = mog.number_of_gaussians();
  num_points = data.n_cols();
  dim = data.n_rows();
  phi_x.Init(num_gauss, num_points);
  weights.Copy(mog.omega()); 
  x.Init(data.n_rows());
  identity_vector.Init(num_points);
  identity_vector.SetAll(1);
  g_mu.Init(num_gauss);
  g_sigma.Init(num_gauss);
	
  for(index_t k = 0; k < num_gauss; k++) {
    g_mu[k].Init(dim);
    g_mu[k].SetZero();
    g_sigma[k].Init((dim * (dim+1) / 2));
    g_sigma[k].SetZero();
    for(index_t i = 0; i < num_points; i++) {
      Vector tmp_g_mu, tmp_g_sigma;
      x.CopyValues(data.GetColumnPtr(i));
      tmpVal = phi(x, mog.mu(k), mog.sigma(k), mog.d_sigma(k), &tmp_g_mu, &tmp_g_sigma);
      phi_x.set(k, i, tmpVal);
      la::AddTo(tmp_g_mu, &g_mu[k]); // calculating the vector sums in g_mu
      la::AddTo(tmp_g_sigma, &g_sigma[k]); // calculating the vector sums in g_sigma
    }
    la::Scale(weights.get(k), &g_mu[k]); // the final scaling of the g_mu
    la::Scale(weights.get(k), &g_sigma[k]); // the final scaling of rhe g_sigma
  }

  la::MulInit(weights, phi_x, &y);
  fit = la::Dot(y, identity_vector); 
  	
  // Calculating the g_omega
  la::MulInit(phi_x, identity_vector, &tmp_g_omega);
  la::MulInit(mog.d_omega(), tmp_g_omega, &g_omega);
	
  // Making the single gradient vector of size K*(D+1)*(D+2)/2
  double *tmp_g_fit;
  tmp_g_fit = (double*)malloc(((num_gauss * (dim+1)*(dim+2) / 2) - 1)*sizeof(double));
  index_t j = 0;
  for(index_t k = 0; k < g_omega.length(); k++)
    tmp_g_fit[k] = g_omega.get(k);
  j = g_omega.length();
  for(index_t k = 0; k < num_gauss; k++){
    for(index_t i = 0; i < dim; i++){
      tmp_g_fit[j + k*dim + i] = g_mu[k].get(i);
    }
    for(index_t i = 0; i < (dim * (dim+1) / 2); i++){
      tmp_g_fit[j + num_gauss*dim + k*(dim * (dim+1) / 2) + i] = g_sigma[k].get(i);
    }
  }
  (*g_fit).Copy(tmp_g_fit, ((num_gauss*(dim+1)*(dim+2) / 2) - 1));
		
  return fit;
}

bool polytope(double **p, long double y[], index_t ndim, long double ftol,
	      long double (*funk)(Matrix&, index_t, double*),index_t *nfunk, 
	      Matrix& data, index_t k_comp) {

  /* funk(x) is the function to be minimized where 'x' is a 'ndim' dimensional point. 
     'p' is 'ndim+1' vertices of the simplex and 'y' is the function value at those 'ndim+1' vertices
     'ftol' is the fractional convergence tolerance to be achieved and 'nfunk' is the number of function evaluations taken */

  index_t i, j, ihi, ilo, inhi, gen_m, mpts = ndim + 1, NMAX = 50000;
  double sum, swap, *psum;
  long double swap_y, rtol, ytry, ysave, TINY = 1.0e-10;
  bool cvgd, TRUE = 1, FALSE = 0;

	
  psum = (double*)malloc(ndim * sizeof(double));
  *nfunk = 0;
  
  gen_m = 0;
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
    if( rtol < ftol ) {
      swap_y = y[0];
      y[0] = y[ilo];
      y[ilo] = swap_y;
      for( i = 0; i < ndim; i++ ) {
	swap = p[0][i];
	p[0][i] =  p[ilo][i] ;
	p[ilo][i] = swap;
      }
      cvgd = TRUE;
      break;
    }
    if(*nfunk > NMAX){
      cvgd = FALSE;
      break;
    }
    *nfunk += 2;
		
    /* Beginning a new iteration. Extrapolating by a factor of -1.0 through the face of the simplex
       across from the high point, i.e, reflect the simplex from the high point */
    if( gen_m >= 50 ) {
      gen_m = 0 ;
      printf(".");
      fflush(NULL);
    }
    else {
      gen_m++;
    }

    for( j = 0 ; j < ndim ; j++ ){
      sum = 0.0;
      for( i = 0 ; i < mpts ; i++ ) 
	if (i != ihi)// added here to remove the highest point from the consideration
	  sum += p[i][j];
      //psum[j] = sum; //removed because we are trying to find the center of gravity of the rest of the points
      psum[j] = sum / ndim;
    }

    ytry = mod_simplex(p, y, psum, ndim, funk, ihi, -1.0, data, k_comp);
    if( ytry <= y[ilo] ) {	// result better than best point so additional extrapolation by a factor of 2
      ytry = mod_simplex(p, y, psum, ndim, funk, ihi, 2.0, data, k_comp);
    }
    else if( ytry >= y[ihi] ) { // result worse than the worst point so there is a lower intermediate point, i.e., do a one dimensional contraction
      ysave = y[ihi];
      //ytry = mod_simplex(p, y, psum, ndim, funk, ihi, 0.5, data, k_comp); // since it is contracting and removing 'ihi' should be -0.5 instead of 0.5
      ytry = mod_simplex(p, y, psum, ndim, funk, ihi, -0.5, data, k_comp);
      if( ytry > y[ihi] ) { // Can't get rid of the high point, try to contract around the best point
	for( i = 0; i < mpts; i++ ) {
	  if( i != ilo ) {
	    for( j = 0; j < ndim; j++ ) 
	      p[i][j] = psum[j] = 0.5 * ( p[i][j] + p[ilo][j] );
	    y[i] = (*funk)(data, k_comp, psum);
	  }
	}
	*nfunk += ndim;
	for( j = 0 ; j < ndim ; j++ ){
	  sum = 0.0;
	  for( i = 0 ; i < mpts ; i++ )
	    if (i != ihi)//same as above
	      sum += p[i][j];
	  //psum[j] = sum; // same as above
	  psum[j] = sum / ndim;
	}
      }
    }
    else --(*nfunk);
  }
  return cvgd;
}

long double mod_simplex(double **p, long double y[], double psum[], index_t ndim,
			long double (*funk)(Matrix&, index_t, double*),index_t ihi,
			float fac, Matrix& data, index_t k_comp) {
	
  /* Extropolates by a factor of 'fac' through the face of the simplex across from the worst point, tries it,
     and replaces the worst point if the new point is better than the worst point.*/
  index_t j;
  //float fac1, fac2; // just trying something
  long double ytry;
  double *ptry;
	
  //printf("mod_simplex:entry\n");
  ptry = (double*) malloc (ndim * sizeof(double));
  //fac1 = (1.0 - fac) / ndim; // removing this because the division by 'ndim' is already done
  // fac1 = (1.0 - fac);
  // fac2 = fac1 - fac;
  for (j = 0; j < ndim; j++) 
    //ptry[j] = psum[j] * fac1 + p[ihi][j] * fac2;
    ptry[j] = psum[j] * (1 - fac) + p[ihi][j] * fac;
  ytry = (*funk)(data, k_comp, ptry);
  if (ytry < y[ihi]) {
    y[ihi] = ytry;
    for (j = 0; j < ndim; j++) {
      //psum[j] += ptry[j] - p[ihi][j];// apparently not needed
      p[ihi][j] = ptry[j];
    }
  }
  return ytry;
}

void quasi_newton(double *p, index_t n, double gradient_tolerance,
		  index_t *number_of_iterations, long double *f_min,
		  long double(*func)(Matrix&, index_t, double*, Vector*),
		  Matrix& data, index_t k_comp) {

  index_t check, i, its, j, MAXIMUM_ITERATIONS = 200;
  long double temp_1, temp_2, temp_3, temp_4, f_previous, 
      maximum_step_length, sum = 0.0, sumdg, sumxi, temp, test;
  Vector dgrad, grad,hdgrad, pold, pnew, xi;
  Matrix hessian;
  long double EPSILON = 3.0e-8;
  long double TOLERANCE = 4*EPSILON;
  double MAX_STEP_SIZE = 100.0;

  dgrad.Init(n);
  grad.Init(n);
  hdgrad.Init(n);
  hessian.Init(n,n);
  pnew.Init(n);
  xi.Init(n);
  pold.Copy(p,n);
  f_previous = (*func)(data, k_comp, p, &grad);
  Vector tmp;
  tmp.Init(n);
  tmp.SetAll(1.0);
  hessian.SetDiagonal(tmp);
  la::ScaleOverwrite(-1.0, g, &xi);
  
  sum = la::Dot(pold, pold);
  double fmax;
  if( sqrt(sum) > (float)n ) fmax = sqrt(sum);
  else { fmax = (float)n; }
  maximum_step_length = MAX_STEP_SIZE*fmax;

  for(its = 0; its < MAXIMUM_ITERATIONS; its++) {
    *number_of_iterations = its;
    dgrad.CopyValues(g);
    line_search(n, pold, f_previous, &grad, &xi,
        &pnew, f_min, maximum_step_length, &check, 
        func, data, k_comp);
    f_previous = *f_min;
    la::SubOverwrite(pold, pnew, &xi);
    pold.CopyValues(pnew);

    test = 0.0;
    for(i = 0; i < n; i++){
      if(fabs(pold.get(i)) > 1.0) fmax = fabs(pold.get(i));
      else{ fmax = 1.0; }
      temp = fabs(xi.get(i)) / fmax;
      if(temp > test) test = temp;
    }
    if(test < TOLERANCE) {
      for(i = 0; i < n; i++) {
	p[i] = pold.get(i);
      }
      return;
    }

    test = 0.0;
    if((*f_min) > 1.0) temp_1 = *f_min;
    else{ temp_1 = 1.0; }

    for(i = 0; i < n; i++) {
      if(fabs(pold.get(i)) > 1.0) fmax = pold.get(i);
      else{ fmax = 1.0; }

      temp = fabs(g.get(i))*fmax / temp_1;
      if(temp > test) test = temp;
    }
    if(test < gradient_tolerance) {
      for(i = 0; i < n; i++) {
	p[i] = pold.get(i);
      }
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
    la::MulOverwrite(hessian, g, &xi);
    la::Scale((-1.0), &xi);
  }
  printf("too many iterations in Quasi Newton\n");
}


void line_search(int n, Vector pold, long double fold, Vector *grad,
		 Vector *xi, Vector *pnew, long double *f_min, 
		 long double maximum_step_length, int *check,
		 long double (*funk)(Matrix&, index_t, double*, Vector*),
		 Matrix& data, index_t k_comp) {

  index_t i;
  long double a, step_length, previous_step_length, 
      minimum_step_length, b, disc, previous_f_value,
      rhs1, rhs2, slope, sum, temp, test, temp_step_length,
      MIN_DECREASE = 1.0e-4, TOLERANCE = 1.0e-7;

  *check = 0;
  sum = la::Dot(*xi, *xi);
  sum = sqrt(sum);
  if(sum > maximum_step_length) {
    la::Scale((maximum_step_length/sum), xi);
  }
  slope = la::Dot(*grad, *xi);
  if(slope >= 0.0){
    printf("blah ");
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

    (*pnew).CopyValues(pold);
    la::AddExpert(step_length, *xi, pnew);
    double *temp_array;
    temp_array = (double*) malloc (n * sizeof(double));
    for(i = 0; i < n; i++) {
      temp_array[i] = (*pnew).get(i);
    }
    *f_min = (*funk)(data, k_comp, temp_array, grad);
    if(step_length < minimum_step_length) {
      (*pnew).CopyValues(pold);
      *check = 1;
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
    step_length = (temp_step_length > 0.1*step_length ? temp_step_length : 0.1*step_length);
  }
}
