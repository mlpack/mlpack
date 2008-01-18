/**
  * Description of the L2 error function we are trying to minimize
  *
  *
  */

#include "fastlib/fastlib.h"
#include "mog.h"
#include "phi.h"

long double l2_error(Matrix&, index_t, double*);

long double l2_error(Matrix&, index_t, double*, Vector*);

long double calc_reg(MoG&);

long double calc_reg(MoG&, Vector*);

long double calc_fit(Matrix&, MoG&);

long double calc_fit(Matrix&, MoG&, Vector*);

long double mod_simplex(double**, long double[], double[], index_t, long double (*funk)(Matrix&, index_t, double*),index_t, float, Matrix&, index_t) ;

bool polytope(double**, long double*, index_t ndim, long double ftol, long double (*funk)(Matrix&, index_t, double*),index_t*, Matrix& data, index_t k_comp);

//void line_search(Vector, long double, Vector*, Vector*, long double*, double, long double (*func) (Matrix&, index_t, double*, Vector*), Matrix&, index_t, Vector*);

//void line_search(Vector, long double, Vector*, Vector*, long double*, double, long double (*func) (double*, Vector*), Vector*);

void line_search(index_t, Vector, long double, Vector, Vector, Vector, long double*, long double, index_t*, long double (*funk)(double*, Vector*));

//void quasi_newton(double*, index_t dim, float grad_tol, index_t*, long double*, long double (*func)(Matrix&, index_t, double*, Vector*), Matrix&, index_t);

void quasi_newton(double*, index_t dim, float grad_tol, index_t*, long double*, long double (*func)(double*, Vector*));

long double l2_error(Matrix& d, index_t num_gauss, double* theta) {
  MoG mog;
  long double reg, fit, l2e; 
  index_t dim, number_of_points;
	
  //printf("l2_error:entry\n");
  number_of_points = d.n_cols();
  dim = d.n_rows();
  mog.MakeModel(num_gauss, dim, theta);
  //M.display();
  //printf("l2_error:1\n");
  reg = calc_reg(mog);
  fit = calc_fit(d, mog);
  //printf("l2_error: reg = %Lf , fit = %Lf \n", reg, fit );
  l2e = reg - 2 * fit / number_of_points ;
  //printf("l2_error:exit\n");
  return l2e;
}
	

long double l2_error(Matrix& d, index_t num_gauss, double* theta, Vector *g_l2_error) {
  MoG mog;
  long double reg, fit, l2e; 
  index_t dim, number_of_points;
  Vector g_reg,g_fit,tmp_fit;
	
  //printf("l2_error:entry\n");
  number_of_points = d.n_cols();
  dim = d.n_rows();
  mog.MakeModelWithGradients(num_gauss, dim, theta);
  //printf("model made\n");
  //fflush(NULL);
  //mog.display();
  //printf("l2_error:1\n");
  reg = calc_reg(mog, &g_reg);
  //printf("reg done\n");
  //fflush(NULL);
  fit = calc_fit(d,mog, &g_fit);
  //  printf("l2_error: reg = %Lf , fit = %Lf \n", reg, fit );
  l2e = reg - 2*fit / number_of_points ;
  double alpha = -2.0 / number_of_points;
  la::ScaleInit(alpha, g_fit, &tmp_fit);
  la::SubInit(tmp_fit, g_reg, g_l2_error);
  //printf("l2_error:exit\n");
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

  //printf("reg entry\n");	
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
  //printf("exit 1\n");
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
  //printf("fit = %Lf\n",fit);
	
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

bool polytope(double **p, long double y[], index_t ndim, long double ftol, long double (*funk)(Matrix&, index_t, double*),index_t *nfunk, Matrix& data, index_t k_comp) {

  /* funk(x) is the function to be minimized where 'x' is a 'ndim' dimensional point. 
     'p' is 'ndim+1' vertices of the simplex and 'y' is the function value at those 'ndim+1' vertices
     'ftol' is the fractional convergence tolerance to be achieved and 'nfunk' is the number of function evaluations taken */

  index_t i, j, ihi, ilo, inhi, gen_m, mpts = ndim + 1, NMAX = 50000;
  double sum, swap, *psum;
  long double swap_y, rtol, ytry, ysave, TINY = 1.0e-10;
  bool cvgd, TRUE = 1, FALSE = 0;

	
  //printf("polytope:entry\n");
  
  psum = (double*)malloc(ndim * sizeof(double));
  *nfunk = 0;
  /*
  for( j = 0 ; j < ndim ; j++ ){
    sum = 0.0;
    for( i = 0 ; i < mpts ; i++ ) 
      sum += p[i][j];
    psum[j] = sum;
  }
  */
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
      //printf("polytope: nfunk = %"LI"d, rtol = %Lf , ftol = %Lf ylo = %Lf\n",*nfunk,rtol,ftol,y[0]);
      cvgd = TRUE;
      break;
    }
    if(*nfunk > NMAX){
      //printf("polytope:maximum number of function evaluations exceeded with ylo = %Lf\n", y[ilo]);
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
  //printf("polytope:exit\n");
  return cvgd;
}

long double mod_simplex(double **p, long double y[], double psum[], index_t ndim, long double (*funk)(Matrix&, index_t, double*),index_t ihi, float fac, Matrix& data, index_t k_comp) {
	
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
  //printf("mod_simplex:exit\n");
  return ytry;
}

//void quasi_newton(double *init_p, index_t dim, float grad_tol, index_t *num_iters, long double *f_min, long double (*func)(Matrix&, index_t, double*, Vector*), Matrix& data, index_t k_comp) {
/*
void quasi_newton(double *init_p, index_t dim, float grad_tol, index_t *num_iters, long double *f_min, long double (*func)(double*, Vector*)) {

  Vector dgrad, grad, hgrad, new_p, old_p, xi, identity_vector;
  //  double* tmp_array;
  Matrix hess;
  double den, stp_max, sum = 0.0, temp, test;
  long double f_val, EPSILON = 3.0e-8, TOL, STEP_LNGTH_MAX = 100.0;
  index_t MAXITER = 2000;

  TOL = 4 * EPSILON;
  
  // Initialize all the vectors and matrix here, too lazy to do it right now 
  dgrad.Init(dim);
  hgrad.Init(dim);
  new_p.Init(dim);
  hess.Init(dim, dim);
  identity_vector.Init(dim);
  identity_vector.SetAll(1.0);
  old_p.Copy(init_p, dim);
  //printf("func start\n");
  //fflush(NULL);
  //for(index_t i = 0; i < dim; i++)
  //printf("%lf ",init_p[i]);

  //f_val = (*func)(data, k_comp, init_p, &grad);
  grad.Init(dim);
  f_val = (*func)(init_p, &grad);
  //  for(index_t i = 0; i < dim; i++)
  //printf("%lf ",grad.get(i));

  //printf("%Lf\n", f_val);
  //printf("func done\n");
  //return;
  //fflush(NULL);
  hess.SetDiagonal(identity_vector);
  la::ScaleInit(-1.0, grad, &xi);
  sum = la::Dot(old_p, old_p);

  stp_max = STEP_LNGTH_MAX * (sqrt(sum) > dim ? sqrt(sum) : dim);

  for (index_t iter = 0; iter < MAXITER; iter++) {
    *num_iters = iter;
    dgrad.CopyValues(grad);
    //line_search(old_p, f_val, &xi, &new_p, f_min, stp_max, func, data, k_comp, &grad );// to be done--- remember to add the gradient value which is calculated at 'new_p'
    line_search(old_p, f_val, &xi, &new_p, f_min, stp_max, func, &grad);
    if((iter % 10) == 0) printf(".");
    fflush(NULL);
    f_val = *f_min;
    la::SubOverwrite(old_p, new_p, &xi);
    old_p.CopyValues(new_p);
    init_p = old_p.ptr();

    test = 0.0;
    for (index_t i = 0; i < dim; i++) {
      temp = fabs(xi.get(i)) / (fabs(old_p.get(i)) > 1.0 ? fabs(old_p.get(i)) : 1.0);
      if (temp > test) 
	test = temp;
    }
    if (test < TOL)
      return;
    
    test = 0.0;
    den = (*f_min > 1.0 ? *f_min : 1.0);
    for(index_t i = 0; i < dim; i++) {
      temp = fabs(grad.get(i)) * (fabs(new_p.get(i)) > 1.0 ? fabs(new_p.get(i)) : 1.0) / den;
      if (temp > test)
	test = temp;
    }
    if (test < grad_tol)
      return;

    la::SubFrom(grad, &dgrad);
    la::Scale(-1.0, &dgrad);
    la::MulOverwrite(hess, dgrad, &hgrad);

    double temp_dot = la::Dot(dgrad, xi);
    double temp_dot_2 = la::Dot(dgrad, hgrad);
    
    if (temp_dot > sqrt(EPSILON * (la::Dot(dgrad, dgrad)) * (la::Dot(xi, xi)))) {
      //dgrad.SetAll(0.0);
      la::ScaleOverwrite((1.0 / temp_dot), xi, &dgrad); // dgrad = alpha * xi -- the function required may also be AddTo
      la::AddExpert((-1.0 / temp_dot_2), hgrad, &dgrad); // dgrad = dgrad + alpha * hdg

      hess.SetAll(0.0);
      Matrix co, ro;
      co.AliasColVector(xi);
      ro.AliasRowVector(xi);
      la::MulOverwrite(co, ro, &hess);
      la::Scale((1.0 / temp_dot), &hess);

      //destruct co, ro somehow
      Matrix co_1, ro_1, temp_matrix_1;
      co_1.AliasColVector(hgrad);
      ro_1.AliasRowVector(hgrad);
      la::MulInit(co_1, ro_1, &temp_matrix_1);
      la::AddExpert((-1.0 / temp_dot_2), temp_matrix_1, &hess);

      //destruct co, ro, temp_matrix somehow
      Matrix co_2, ro_2, temp_matrix_2;
      co_2.AliasColVector(dgrad);
      ro_2.AliasRowVector(dgrad);
      la::MulInit(co_2, ro_2, &temp_matrix_2);
      la::AddExpert(temp_dot_2, temp_matrix_2, &hess);
    }
    la::MulOverwrite(hess, grad, &xi);
    la::Scale(-1.0, &xi);
  }

  printf("Quasi - Newton: Exceeded the maximum allowed number of iterations\n");
}
*/
/*
//void line_search(Vector old_p, long double f_old, Vector *direction, Vector *new_p, long double *f_new, double max_step_length, long double (*func) (Matrix&, index_t, double*, Vector*), Matrix& data, index_t k_comp, Vector *grad) {
void line_search(Vector old_p, long double f_old, Vector *direction, Vector *new_p, long double *f_new, double max_step_length, long double (*func) (double*, Vector*), Vector *grad) {

  //  double* temp_array;
  //temp_array = (double*) malloc (old_p.length() * sizeof(double));
  long double TOL = 1.0e-7, MIN_DECREASE = 1.0e-4;
  double tmp_dot = la::Dot(*direction, *direction);
  if (sqrt(tmp_dot) > max_step_length)
    la::Scale((max_step_length / sqrt(tmp_dot)), direction);

  double slope = la::Dot(*grad, *direction);
  if (slope >= 0.0)
    fprintf(stderr,"line search: Roundoff problem\n");

  double temp, test = 0.0;
  for(index_t i = 0; i < old_p.length(); i++) {
    temp = fabs((*direction).get(i)) / (fabs(old_p.get(i)) > 1.0 ? fabs(old_p.get(i)) : 1.0);
    if (temp > test)
      test = temp;
  }
  double min_step_length = TOL / test; //check this line properly
  double prev_step_length, step_length = 1.0;
  double temp_step_length, temp1, temp2, temp3, temp4;
  long double prev_f_new;

  for(;;) {

    (*new_p).CopyValues(old_p);
    la::AddExpert(step_length, (*direction), new_p);
    //for(index_t i = 0; i < (*new_p).length(); i++){
    //temp_array[i] = (*new_p).get(i);
    //printf("%lf ",temp_array[i]);
    //}
    //(*f_new) = (*func) (data, k_comp, (*new_p).ptr(), grad);
    (*f_new) = (*func) ((*new_p).ptr(), grad);
    //printf(" val = %Lf\n", *f_new);
    if (step_length < min_step_length) {
      (*new_p).CopyValues(old_p);
      return;
    }
    else if (*f_new <= f_old + MIN_DECREASE * step_length * slope)
      return;
    else {
      if (step_length == 1.0)
	temp_step_length = -slope / (2.0 * (*f_new - f_old - slope)) ;
      else {
	temp1 = *f_new - f_old - step_length * slope;
	temp2 = prev_f_new - f_old - prev_step_length * slope;
	temp3 = (temp1 / (step_length * step_length) - temp2 / (prev_step_length * prev_step_length)) / (step_length - prev_step_length);
	temp4 = (-prev_step_length * temp1 / (step_length * step_length) + step_length * temp2 / (prev_step_length * prev_step_length)) / (step_length - prev_step_length);
	if (temp3 == 0.0)
	  temp_step_length = -slope / (2.0 * temp4);
	else {
	  double temp5 = temp4 * temp4 - 3.0 * temp3 * slope;
	  if (temp5 < 0.0)
	    temp_step_length = 0.5 * step_length;
	  else if (temp4 <= 0.0)
	    temp_step_length = (-temp4 + sqrt(temp5)) / (3.0 * temp3);
	  else
	    temp_step_length = -slope / (temp4 + sqrt(temp5));
	}
	if (temp_step_length > 0.5 * step_length)
	  temp_step_length = 0.5 * step_length;
      }
    }
    prev_step_length = step_length;
    prev_f_new = *f_new;
    step_length = (temp_step_length > 0.1 * step_length ? temp_step_length : 0.1 * step_length);
  }
}
*/
void quasi_newton(double *p, index_t n, double gtol, index_t *iter, long double *fret, long double(*func)(double*, Vector*)) {

  index_t check, i, its, j, ITMAX = 200;
  long double den, fac,fad,fae,fp,stpmax, sum = 0.0, sumdg, sumxi, temp, test;
  Vector dg, g,hdg, pold, pnew, xi;
  Matrix hessin;
  long double EPS = 3.0e-8;
  long double TOLX = 4*EPS;
  double STPMX = 100.0;

  dg.Init(n);
  g.Init(n);
  hdg.Init(n);
  hessin.Init(n,n);
  pnew.Init(n);
  xi.Init(n);
  pold.Copy(p,n);
  fp = (*func)(p, &g);
  Vector tmp;
  tmp.Init(n);
  tmp.SetAll(1.0);
  hessin.SetDiagonal(tmp);
  la::ScaleOverwrite(-1.0, g, &xi);
  for(i = 0; i < n; i++) {
    printf("%lf ", xi.get(i));
  }
  printf("\n");
  sum = la::Dot(pold, pold);

  double fmax;
  if( sqrt(sum) > (float)n ) fmax = sqrt(sum);
  else { fmax = (float)n; }
  stpmax = STPMX*fmax;

  for(its = 0; its < ITMAX; its++) {
    *iter = its;
    dg.CopyValues(g);
    line_search(n, pold, fp, g, xi, pnew, fret, stpmax, &check, func);
    fp = *fret;
    la::SubOverwrite(pold, pnew, &xi);
    pold.CopyValues(pnew);

    test = 0.0;
    for(i = 0; i < n; i++){
      if(fabs(pold.get(i)) > 1.0) fmax = fabs(pold.get(i));
      else{ fmax = 1.0; }
      temp = fabs(xi.get(i)) / fmax;
      if(temp > test) test = temp;
    }
    if(test < TOLX) {
      return;
    }

    test = 0.0;
    if((*fret) > 1.0) den = *fret;
    else{ den = 1.0; }

    for(i = 0; i < n; i++) {
      if(fabs(pold.get(i)) > 1.0) fmax = pold.get(i);
      else{ fmax = 1.0; }

      temp = fabs(g.get(i))*fmax / den;
      if(temp > test) test = temp;
    }
    if(test < gtol) {
      return;
    }

    la::SubFrom(g,&dg);
    la::Scale(-1.0, &dg);
    la::MulOverwrite(hessin,dg, &hdg);

    fac = la::Dot(dg, xi);
    fae = la::Dot(dg, hdg);
    sumdg = la::Dot(dg, dg);
    sumxi = la::Dot(xi, xi);

    if (fac > sqrt(EPS*sumdg*sumxi)) {
      fac = 1.0 / fac;
      fad = 1.0 / fae;

      la::ScaleOverwrite(fac, xi, &dg);
      la::AddExpert((-1.0*fad), hdg, &dg);

      Matrix co, ro, tmp;
      co.AliasColVector(xi);
      ro.AliasRowVector(xi);
      la::MulInit(co, ro, &tmp);
      la::AddExpert(fac, tmp, &hessin);

      //      la::Scale((1.0 / temp_dot), &hess);

      co.Destruct();
      ro.Destruct();
      tmp.Destruct();

      //      Matrix co_1, ro_1, temp_matrix_1;
      co.AliasColVector(hdg);
      ro.AliasRowVector(hdg);
      la::MulInit(co, ro, &tmp);
      la::AddExpert((-1.0*fad), tmp, &hessin);

      //destruct co, ro, temp_matrix somehow
      co.Destruct();
      ro.Destruct();
      tmp.Destruct();

      //     Matrix co_2, ro_2, temp_matrix_2;
      co.AliasColVector(dg);
      ro.AliasRowVector(dg);
      la::MulInit(co, ro, &tmp);
      la::AddExpert(fae, tmp, &hessin);
    }
    la::MulOverwrite(hessin, g, &xi);
    la::Scale((-1.0), &xi);
  }
  printf("too many iterations in Quasi Newton\n");
}


void line_search(int n, Vector pold, long double fold, Vector g, Vector xi, Vector pnew, long double *fret, long double stpmax, int *check, long double (*funk)(double*, Vector*)) {
  index_t i;
  long double a, alam, alam2, alamin, b, disc, f2, rhs1, rhs2, slope, sum, temp, test, tmplam, ALF = 1.0e-4, TOLX = 1.0e-7;

  *check = 0;
  sum = la::Dot(xi, xi);
  sum = sqrt(sum);
  if(sum > stpmax) {
    la::Scale((stpmax/sum), &xi);
  }
  slope = la::Dot(g, xi);
  if(slope >= 0.0){
    printf("blah ");
    return;
  }
  test = 0.0;
  for(i = 0; i < n; i++) {
    double fmax;
    fmax = (fabs(pold.get(i)) > 1.0 ? fabs(pold.get(i)) : 1.0);
    temp = fabs(xi.get(i)) / fmax;
    if(temp > test) test = temp;
  }

  alamin = TOLX/test;
  alam = 1.0;
  for(;;) {

    pnew.CopyValues(pold);
    la::AddExpert(alam, xi, &pnew);
    double *temp_array;
    temp_array = (double*) malloc (n * sizeof(double));
    for(i = 0; i < n; i++) {
      temp_array[i] = pnew.get(i);
    }
    *fret = (*funk)(temp_array, &g);
    if(alam < alamin) {
      pnew.CopyValues(pold);
      *check = 1;
      return;
    }
    else if( *fret <= fold + ALF*alam*slope) {
      return;
    }
    else {
      if (alam == 1.0) {
	tmplam = -slope/(2.0*(*fret - fold - slope));
      }
      else {
	rhs1 = *fret - fold - alam*slope;
	rhs2 = f2 - fold - alam2*slope;
	a = (rhs1 / (alam*alam) - rhs2/(alam2*alam2))/(alam-alam2);
	b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2)) / (alam - alam2);
	if(a == 0.0) {
	  tmplam = -slope / (2.0*b);
	}
	else {
	  disc = b*b - 3.0*a*slope;
	  if(disc < 0.0) {
	    tmplam = 0.5*alam;
	  }
	  else if (b <= 0.0) {
	    tmplam = (-b+sqrt(disc))/(3.0*a);
	  }
	  else {
	    tmplam = -slope / (b+sqrt(disc));
	  }
	}
	if(tmplam > 0.5*alam) {
	  tmplam = 0.5*alam;
	}
      }
    }
    alam2 = alam;
    f2 = *fret;
    alam = (tmplam > 0.1*alam ? tmplam : 0.1*alam);
  }
}

