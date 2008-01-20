/**
 * @file mog.cc
 *
 * Implementation for the loglikelihood function, the EM algorithm,
 * the L2 estimation, computes the K-means for getting an initial point
 *
 */

#include "mog.h"
#include "phi.h"
#include "math_functions.h"
#include "l2_error.h"


void MoG::ExpectationMaximization(Matrix& data_points, ArrayList<double> *results) {
  
  // Declaration of the variables
  index_t num_points;
  index_t dim, num_gauss;
  double sum, tmp; 
  ArrayList<Vector> mu_temp;
  ArrayList<Matrix> sigma_temp;
  Vector omega_temp, x;
  Matrix cond_prob;	
  long double l, l_old, best_l, INFTY = 99999, TINY = 1.0e-10;

  // Initializing values
  dim = dimension();
  num_gauss = number_of_gaussians();
  num_points = data_points.n_cols();

  // Initializing the number of the vectors and matrices 
  // according to the parameters input
  mu_temp.Init(num_gauss);
  sigma_temp.Init(num_gauss);
  omega_temp.Init(num_gauss);
  
  // Allocating size to the vectors and matrices
  // according to the dimensionality of the data
  for(index_t i = 0; i < num_gauss; i++) {
    mu_temp[i].Init(dim);
    sigma_temp[i].Init(dim, dim);    
  }
  x.Init(dim);
  cond_prob.Init(num_gauss, num_points);
  
  best_l = -INFTY;
  index_t restarts = 0;
  // performing 5 restarts and choosing the best from them
  while (restarts < 5) { 

    // assign initial values to 'mu', 'sig' and 'omega' using k-means
    KMeans(data_points, &mu_temp, &sigma_temp, &omega_temp, num_gauss);
    
    l_old = -INFTY;

    // calculates the loglikelihood value
    l = Loglikelihood(data_points, mu_temp, sigma_temp, omega_temp); 
 
    // added a check here to see if any 
    // significant change is being made 
    // at every iteration
    while (l - l_old > TINY) {
      // calculating the conditional probabilities 
      // of choosing a particular gaussian given 
      // the data and the present theta value
      for (index_t j = 0; j < num_points; j++) {
	x.CopyValues(data_points.GetColumnPtr(j));
	sum = 0;
	for (index_t i = 0; i < num_gauss; i++) {
	  tmp = phi(x, mu_temp[i], sigma_temp[i]) * omega_temp.get(i);
	  cond_prob.set(i, j, tmp);
	  sum += tmp;	  
	}
	for (index_t i = 0; i < num_gauss; i++) {
	  tmp = cond_prob.get(i, j);
	  cond_prob.set(i, j, tmp / sum); 
	}
      }
			
      // calculating the new value of the mu 
      // using the updated conditional probabilities
      for (index_t i = 0; i < num_gauss; i++) {
	sum = 0;
	mu_temp[i].SetZero();
	for (index_t j = 0; j < num_points; j++) {
	  x.CopyValues(data_points.GetColumnPtr(j));
	  la::AddExpert(cond_prob.get(i, j), x, &mu_temp[i]);
	  sum += cond_prob.get(i, j); 
	}
	la::Scale((1.0 / sum), &mu_temp[i]);
      }
					
      // calculating the new value of the sig 
      // using the updated conditional probabilities
      // and the updated mu
      for (index_t i = 0; i < num_gauss; i++) {
	sum = 0;
	sigma_temp[i].SetZero();
	for (index_t j = 0; j < num_points; j++) {
	  Matrix co, ro, c;
	  c.Init(dim, dim);
	  x.CopyValues(data_points.GetColumnPtr(j));
	  la::SubFrom(mu_temp[i] , &x);
	  co.AliasColVector(x);
	  ro.AliasRowVector(x);
	  la::MulOverwrite(co, ro, &c);
	  la::AddExpert(cond_prob.get(i, j), c, &sigma_temp[i]);
	  sum += cond_prob.get(i, j);	  
	}
	la::Scale((1.0 / sum), &sigma_temp[i]);
      }
			
      // calculating the new values for omega 
      // using the updated conditional probabilities
      Vector identity_vector;
      identity_vector.Init(num_points);
      identity_vector.SetAll(1.0 / num_points);
      la::MulOverwrite(cond_prob, identity_vector, &omega_temp);
      
      l_old = l;
      l = Loglikelihood(data_points, mu_temp, sigma_temp, omega_temp);
    }
    
    // putting a check to see if the best one is chosen
    if(l > best_l){
      best_l = l;
      for (index_t i = 0; i < num_gauss; i++) {
	mu_[i].CopyValues(mu_temp[i]);
	sigma_[i].CopyValues(sigma_temp[i]);
	omega_.CopyValues(omega_temp);
      }
    }
    restarts++;
  }	
  
  OutputResults(results);
  return;
}

long double MoG::Loglikelihood(Matrix& data_points, ArrayList<Vector>& means,
			       ArrayList<Matrix>& covars, Vector& weights) {
  index_t i, j;
  Vector x;
  long double likelihood, loglikelihood = 0;
	
  x.Init(data_points.n_rows());
	
  for (j = 0; j < data_points.n_cols(); j++) {
    x.CopyValues(data_points.GetColumnPtr(j));
    likelihood = 0;
    for(i = 0; i < number_of_gaussians() ; i++){
      likelihood += weights.get(i) * phi(x, means[i], covars[i]);
    }
    loglikelihood += log(likelihood);
  }
  return loglikelihood;
}

void MoG::KMeans(Matrix& data, ArrayList<Vector> *means,
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
  (*weights).SetAll(tmp / value_of_k);
  return;
}

void MoG::L2Estimation(Matrix& data, ArrayList<double> *results, int optim_flag) {

  // Declaration of variables
  double *theta;
  double **initPoints;
  long double* initVal;
  double *init_theta;
  long double final_val;
  long double ftol, grad_tol, error_val = 0.0, EPS = 1.0e-7;
  index_t dim_theta, niters, restarts = 0;
  bool cvgd, TRUE = 1, FALSE = 0;

  // Initializing the variables
  index_t num_gauss = number_of_gaussians();
  dim_theta = num_gauss*(dimension() + 1)*(dimension() + 2) / 2 - 1;
  ftol = EPS;
  grad_tol = EPS;

  theta = (double*) malloc (dim_theta * sizeof(double));
  init_theta = (double*) malloc (dim_theta * sizeof(double));
  
  initPoints = (double**) malloc ((dim_theta + 1) * sizeof(double*));
  for(index_t i = 0; i < dim_theta + 1; i++) {
    initPoints[i] = (double*) malloc (dim_theta * sizeof(double));
  }

  initVal = (long double*) malloc ((dim_theta + 1) * sizeof(long double));
  
  // variable to see if the polytope converges
  cvgd = TRUE;
  // calling the polytope optimizer
  if (optim_flag == 1) {
    while (restarts < 5) {
      do {
	// get the 'dim_theta + 1' initial points
	points_generator(data, initPoints, dim_theta + 1, num_gauss);
	
	// obtaining the function values at those points
	for( index_t i = 0 ; i < dim_theta + 1 ; i++ ) {
	  initVal[i] = l2_error( data, num_gauss, initPoints[i]) ;
	}
		
	// The optimizer ...
	cvgd = polytope( initPoints, initVal, dim_theta, 
			 ftol, l2_error, &niters, data, num_gauss );
      }while (!cvgd);
     
      // choosing the best of the 5 restarts 
      if (error_val > initVal[0]) {
	for (index_t i = 0; i < dim_theta; i++)
	  theta[i] = initPoints[0][i];
	error_val = initVal[0];
      }
      restarts++;
    }
  }
    
  // using the quasi newtion optmizer   
  else {
    while (restarts < 5) {
      // get a random starting point
      initial_point_generator(init_theta, data, num_gauss);
      // The optimizer.......
      quasi_newton(init_theta, dim_theta, grad_tol, 
		   &niters, &final_val, l2_error, data, num_gauss);
      
      // choosing the best of the 5 restarts 
      if (error_val > final_val) {
	for (index_t i = 0; i < dim_theta; i++)
	  theta[i] = init_theta[i];
	error_val = final_val;
      }
      restarts++;
    }
  }

  MakeModel(num_gauss, dimension(), theta);
  OutputResults(results);
  return;
}

void MoG::points_generator(Matrix& d, double **points, index_t number_of_points,
			   index_t number_of_components) {

  index_t dim, n, i, j, x;
  
  dim = dimension();
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

void MoG::initial_point_generator (double *theta, Matrix& data, index_t k_comp) {

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

  KMeans(data, &means, &covars, &weights, k_comp);

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
