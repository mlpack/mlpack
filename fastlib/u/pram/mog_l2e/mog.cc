/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog.cc
 *
 * Implementation for L2 loss function, and 
 * also some initial points generator
 *
 */

#include "mog.h"
#include "phi.h"

long double MoGL2E::L2Error(const Matrix& data, Vector *gradients) {

  long double reg, fit, l2e;
  index_t number_of_points = data.n_cols();


  if (gradients != NULL) {
    Vector g_reg, g_fit;
    reg = RegularizationTerm_(&g_reg);
    fit = GoodnessOfFitTerm_(data, &g_fit);  
    DEBUG_ASSERT(gradients->length() == (number_of_gaussians()*
					 (dimension()+1)*(dimension()+2)/2 - 1)
		 
		 );
    gradients->SetAll(0.0);
    la::AddTo(g_reg, gradients);
    la::AddExpert((-2.0/number_of_points), g_fit, gradients);
  }
  else { 
    reg = RegularizationTerm_();
    fit = GoodnessOfFitTerm_(data);
  }
  l2e = reg - 2*fit / number_of_points; 
  return l2e;
}

long double MoGL2E::RegularizationTerm_(Vector *g_reg){
  Matrix phi_mu, sum_covar;
  Vector x, y;
  long double reg, tmpVal;
  index_t num_gauss, dim;

  Vector df_dw, g_omega;
  ArrayList<Vector> g_mu, g_sigma;
  ArrayList<ArrayList<Vector> > dp_d_mu, dp_d_sigma;
  

  num_gauss = number_of_gaussians();
  dim = dimension();
  
  phi_mu.Init(num_gauss, num_gauss);
  sum_covar.Init(dim, dim);
  x.Copy(omega());

  if (g_reg != NULL) {
    g_mu.Init(num_gauss);
    g_sigma.Init(num_gauss);
    dp_d_mu.Init(num_gauss);
    dp_d_sigma.Init(num_gauss);
    for(index_t k = 0; k < num_gauss; k++){
      dp_d_mu[k].Init(num_gauss);
      dp_d_sigma[k].Init(num_gauss);
    }
  }
  else {
    g_mu.Init(0);
    g_sigma.Init(0);
    dp_d_mu.Init(0);
    dp_d_sigma.Init(0);
    df_dw.Init(0);
    g_omega.Init(0);
  }

  for(index_t k = 1; k < num_gauss; k++) {
    for(index_t j = 0; j < k; j++) {
      la::AddOverwrite(sigma(k), sigma(j), &sum_covar);
			
      if (g_reg != NULL) {
	ArrayList<Matrix> tmp_d_cov;
	Vector tmp_dp_d_sigma;
      
	tmp_d_cov.Init(dim*(dim+1));
	for(index_t i = 0; i < (dim*(dim + 1) / 2); i++){
	  tmp_d_cov[i].Copy(d_sigma(k)[i]);
	  tmp_d_cov[(dim*(dim+1)/2)+i].Copy(d_sigma(j)[i]);
	}
      
	//tmpVal = phi(mu(k),mu(j),sum_covar,
	//     tmp_d_cov,&dp_d_mu[j][k],&tmp_dp_d_sigma);
	tmpVal = phi(mu(k),mu(j),sum_covar,tmp_d_cov, &dp_d_mu[k][j],
		     &tmp_dp_d_sigma);
      
	phi_mu.set(j, k, tmpVal);
	phi_mu.set(k, j, tmpVal);
			
	// la::ScaleInit(-1.0, dp_d_mu[j][k], &dp_d_mu[k][j]);
	la::ScaleInit(-1.0, dp_d_mu[k][j], &dp_d_mu[j][k]);
			
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
      else {
	tmpVal = phi(mu(k), mu(j), sum_covar);
	phi_mu.set(j, k, tmpVal);
	phi_mu.set(k, j, tmpVal);
      }
    }
  }
	
  for(index_t k = 0; k < num_gauss; k++) {
    la::ScaleOverwrite(2, sigma(k), &sum_covar);

    if (g_reg != NULL) {
      Vector junk;
      tmpVal = phi(mu(k), mu(k), sum_covar,
		   d_sigma(k), &junk, &dp_d_sigma[k][k]);
      phi_mu.set(k, k, tmpVal);
      dp_d_mu[k][k].Init(dim);
      dp_d_mu[k][k].SetZero();
    }
    else {
      tmpVal = phi(mu(k), mu(k), sum_covar);
      phi_mu.set(k, k, tmpVal);

    }
  }
	
  // Calculating the reg value
  la::MulInit( x, phi_mu, &y );
  reg = la::Dot( x, y );
	
  if (g_reg != NULL) {
    // Calculating the g_omega values - a vector of size K-1
    la::ScaleInit(2.0,y,&df_dw);
    la::MulInit(d_omega(),df_dw,&g_omega);
	
    // Calculating the g_mu values - K vectors of size D
    for(index_t k = 0; k < num_gauss; k++){
      g_mu[k].Init(dim);
      g_mu[k].SetZero();
      for(index_t j = 0; j < num_gauss; j++) {
	la::AddExpert(x.get(j), dp_d_mu[j][k], &g_mu[k]);
      }
      la::Scale((2.0 * x.get(k)), &g_mu[k]);
    }
	
    // Calculating the g_sigma values - K vectors of size D(D+1)/2
    for(index_t k = 0; k < num_gauss; k++){
      g_sigma[k].Init((dim*(dim + 1)) / 2);
      g_sigma[k].SetZero();
      for(index_t j = 0; j < num_gauss; j++) {
	la::AddExpert(x.get(j), dp_d_sigma[j][k], &g_sigma[k]);
      }
      la::Scale((2.0 * x.get(k)), &g_sigma[k]);
    }
	
    // Making the single gradient vector of size K*(D+1)*(D+2)/2 - 1
    double *tmp_g_reg;
    tmp_g_reg = (double*)malloc(((num_gauss*(dim + 1)*(dim + 2) / 2) - 1)
				*sizeof(double));
    index_t j = 0;
    for(index_t k = 0; k < g_omega.length(); k++) {
      tmp_g_reg[k] = g_omega.get(k);
    }
    j = g_omega.length();
    for(index_t k = 0; k < num_gauss; k++){
      for(index_t i = 0; i < dim; i++){
	tmp_g_reg[j + k*(dim) + i] = g_mu[k].get(i);
      }
      for(index_t i = 0; i < (dim*(dim+1)/2); i++){
	tmp_g_reg[j + num_gauss*dim 
		  + k*(dim*(dim+1) / 2) 
		  + i] = g_sigma[k].get(i);
      }
    }
    g_reg->Copy(tmp_g_reg, ((num_gauss*(dim+1)*(dim+2) / 2) - 1));
  }
	
  return reg;
}

long double MoGL2E::GoodnessOfFitTerm_(const Matrix& data, Vector *g_fit) {
  long double fit;
  Matrix phi_x;
  Vector weights, x, y, identity_vector;
  index_t num_gauss, num_points, dim;
  long double tmpVal;
  Vector g_omega,tmp_g_omega;
  ArrayList<Vector> g_mu, g_sigma;
 
  num_gauss = number_of_gaussians();
  num_points = data.n_cols();
  dim = data.n_rows();
  phi_x.Init(num_gauss, num_points);
  weights.Copy(omega()); 
  x.Init(data.n_rows());
  identity_vector.Init(num_points);
  identity_vector.SetAll(1);

  if(g_fit != NULL) {
    g_mu.Init(num_gauss);
    g_sigma.Init(num_gauss);
  }
  else {
    g_mu.Init(0);
    g_sigma.Init(0);
    g_omega.Init(0);
    tmp_g_omega.Init(0);
  }
	
  for(index_t k = 0; k < num_gauss; k++) {
    if (g_fit != NULL) {
      g_mu[k].Init(dim);
      g_mu[k].SetZero();
      g_sigma[k].Init((dim * (dim+1) / 2));
      g_sigma[k].SetZero();
    }
    for(index_t i = 0; i < num_points; i++) {
      if (g_fit != NULL) {
	Vector tmp_g_mu, tmp_g_sigma;
	x.CopyValues(data.GetColumnPtr(i));
	tmpVal = phi(x, mu(k), sigma(k),
		     d_sigma(k), &tmp_g_mu, &tmp_g_sigma);
	phi_x.set(k, i, tmpVal);
	la::AddTo(tmp_g_mu, &g_mu[k]);
	la::AddTo(tmp_g_sigma, &g_sigma[k]);
      }
      else {
	x.CopyValues(data.GetColumnPtr(i));
	phi_x.set(k, i, phi(x, mu(k), sigma(k)));
      }
    }
    if (g_fit != NULL) {
      la::Scale(weights.get(k), &g_mu[k]); 
      la::Scale(weights.get(k), &g_sigma[k]);
    }
  }

  la::MulInit(weights, phi_x, &y);
  fit = la::Dot(y, identity_vector); 
  	
  if (g_fit != NULL) {
    // Calculating the g_omega
    la::MulInit(phi_x, identity_vector, &tmp_g_omega);
    la::MulInit(d_omega(), tmp_g_omega, &g_omega);
	
    // Making the single gradient vector of size K*(D+1)*(D+2)/2
    double *tmp_g_fit;
    tmp_g_fit = (double*)malloc(((num_gauss * (dim+1)*(dim+2) / 2) - 1)
				*sizeof(double));
    index_t j = 0;
    for(index_t k = 0; k < g_omega.length(); k++)
      tmp_g_fit[k] = g_omega.get(k);
    j = g_omega.length();
    for(index_t k = 0; k < num_gauss; k++){
      for(index_t i = 0; i < dim; i++){
	tmp_g_fit[j + k*dim + i] = g_mu[k].get(i);
      }
      for(index_t i = 0; i < (dim * (dim+1) / 2); i++){
	tmp_g_fit[j + num_gauss*dim 
		  + k*(dim * (dim+1) / 2) 
		  + i] = g_sigma[k].get(i);
      }
    }
    g_fit->Copy(tmp_g_fit, ((num_gauss*(dim+1)*(dim+2) / 2) - 1));
  }
  return fit;
}

void MoGL2E::MultiplePointsGenerator(double **points,
				     index_t number_of_points,
				     const Matrix& d,
				     index_t number_of_components) {

  index_t dim, n, i, j, x;
  
  dim = d.n_rows();
  n = d.n_cols();
  
  for( i = 0; i < number_of_points; i++) {
    for(j = 0; j < number_of_components - 1; j++) {
      points[i][j] = (rand() % 20001)/1000 - 10;
    }
  }

  for(i = 0; i < number_of_points; i++){
    for(j = 0; j < number_of_components; j++){
      Vector tmp_mu;
      tmp_mu.Init(dim);
      tmp_mu.CopyValues(d.GetColumnPtr((rand() % n)));
      for(x = 0; x < dim; x++)
	points[i][number_of_components - 1 + j * dim + x] = tmp_mu.get(x);
    }
  } 
 
  for(i = 0; i < number_of_points; i++)
    for(j = 0; j < number_of_components; j++) 
      for(x = 0 ; x < (dim * (dim + 1) / 2); x++)
	points[i][(number_of_components * (dim + 1) - 1) 
		  + (j * (dim * (dim + 1) / 2)) + x] = (rand() % 501)/100;

  return;
}

void MoGL2E::InitialPointGenerator(double *theta, const Matrix& data,
				   index_t k_comp) {

  ArrayList<Vector> means;
  ArrayList<Matrix> covars;
  Vector weights;
  double temp, noise;
  index_t dim;

  weights.Init(k_comp);
  means.Init(k_comp);
  covars.Init(k_comp);
  dim = data.n_rows();

  for (index_t i = 0; i < k_comp; i++) {
    means[i].Init(dim);
    covars[i].Init(dim, dim);
  }

  KMeans_(data, &means, &covars, &weights, k_comp);

  for(index_t k = 0; k < k_comp - 1; k++){
    temp = weights[k] / weights[k_comp - 1];
    noise = (double)(rand() % 10000) / (double)1000;
    theta[k] = noise - 5;
  }
  for(index_t k = 0; k < k_comp; k++){
    for(index_t j = 0; j < dim; j++)
      theta[k_comp - 1 + k * dim + j] = means[k].get(j);

    Matrix U, U_tran;
    la::CholeskyInit(covars[k], &U);
    la::TransposeInit(U, &U_tran);
    for(index_t j = 0; j < dim; j++) {
      for(index_t i = 0; i < j + 1; i++) {
	noise = (rand() % 501) / 100;
	theta[k_comp - 1 + k_comp * dim 
	      + k * dim * (dim + 1) / 2 
	      + j * (j + 1) / 2 + i] = U_tran.get(j, i) + noise;
      }
    }
  }
  return;
}

void MoGL2E::KMeans_(const Matrix& data, ArrayList<Vector> *means,
		     ArrayList<Matrix> *covars, Vector *weights, 
		     index_t value_of_k){
  
  ArrayList<Vector> mu, mu_old;
  double* tmpssq;
  double* sig;
  double* sig_best;
  index_t *y;
  Vector x, diff;
  Matrix ssq;
  index_t i, j, k, n, t, dim;
  double score, score_old, sum;
	
  n = data.n_cols();
  dim = data.n_rows();
  mu.Init(value_of_k);
  mu_old.Init(value_of_k);
  tmpssq = (double*)malloc(value_of_k * sizeof( double ));
  sig = (double*)malloc(value_of_k * sizeof( double ));
  sig_best = (double*)malloc(value_of_k * sizeof( double ));
  ssq.Init(n, value_of_k);
	
  for( i = 0; i < value_of_k; i++){
    mu[i].Init(dim);
    mu_old[i].Init(dim);
  }
  x.Init(dim);
  y = (index_t*)malloc(n * sizeof(index_t));
  diff.Init(dim);
	
  score_old = 999999;
	
  // putting 5 random restarts to obtain the k-means
  for(i = 0; i < 5; i++){
    t = -1;
    for (k = 0; k < value_of_k; k++){
      t = (t + 1 + (rand()%((n - 1 - (value_of_k - k)) - (t + 1))));
      mu[k].CopyValues(data.GetColumnPtr(t));
      for(j = 0; j < n; j++){
	x.CopyValues( data.GetColumnPtr(j));
	la::SubOverwrite(mu[k], x, &diff);
	ssq.set( j, k, la::Dot(diff, diff));
      }      	
    }
    min_element(ssq, y);
    
    do{
      for(k = 0; k < value_of_k; k++){
	mu_old[k].CopyValues(mu[k]);
      }
			
      for(k = 0; k < value_of_k; k++){
	index_t p = 0;
	mu[k].SetZero();
	for(j = 0; j < n; j++){
	  x.CopyValues(data.GetColumnPtr(j));
	  if(y[j] == k){
	    la::AddTo(x, &mu[k]);
	    p++;
	  }
	}
	
	if(p == 0){
	}
	else{
	  double sc = 1 ;
	  sc = sc / p;
	  la::Scale(sc , &mu[k]);
	}
	for(j = 0; j < n; j++){
	  x.CopyValues(data.GetColumnPtr(j));
	  la::SubOverwrite(mu[k], x, &diff);
	  ssq.set(j, k, la::Dot(diff, diff));
	}
      }      
      min_element(ssq, y);
      
      sum = 0;
      for(k = 0; k < value_of_k; k++) {
	la::SubOverwrite(mu[k], mu_old[k], &diff);
	sum += la::Dot(diff, diff);
      }
    }while(sum != 0);
		
    for(k = 0; k < value_of_k; k++){
      index_t p = 0;
      tmpssq[k] = 0;
      for(j = 0; j < n; j++){
	if(y[j] == k){
	  tmpssq[k] += ssq.get(j, k);
	  p++;
	}
      }
      sig[k] = sqrt(tmpssq[k] / p);
    }	
		
    score = 0;
    for(k = 0; k < value_of_k; k++){
      score += tmpssq[k];
    }
    score = score / n;
    
    if (score < score_old) {
      score_old = score;
      for(k = 0; k < value_of_k; k++){
	(*means)[k].CopyValues(mu[k]);
	sig_best[k] = sig[k];	
      }
    }
  }
	
  for(k = 0; k < value_of_k; k++){
    x.SetAll(sig_best[k]);
    (*covars)[k].SetDiagonal(x);
  }
  double tmp = 1;
  weights->SetAll(tmp / value_of_k);
  return;
}

void MoGL2E::min_element( Matrix& element, index_t *indices ){
	
  index_t last = element.n_cols() - 1;
  index_t first, lowest;
  index_t i;
	
  for( i = 0; i < element.n_rows(); i++ ){
		
    first = lowest = 0;
    if(first == last){
      indices[ i ] = last;
    }
    while(++first <= last){
      if( element.get( i , first ) < element.get( i , lowest ) ){
	lowest = first;
      }
    }
    indices[ i ] = lowest;
  }
  return;
}

