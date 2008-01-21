/**
 * @author pram
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

void MoGL2E::KMeans(Matrix& data, ArrayList<Vector> *means,
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

void MoGL2E::L2Estimation(Matrix& data, ArrayList<double> *results, int optim_flag) {

  // Declaration of variables
  double *theta;
  double **initPoints;
  long double* initVal;
  double *init_theta;
  long double final_val;
  long double ftol, grad_tol, error_val = 0.0, EPS = 1.0e-7;
  index_t dim_theta, niters, restarts = 0;
  bool cvgd;
  
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
  cvgd = 0;
  // calling the polytope optimizer
  if (optim_flag == 1) {
    while (restarts < 5) {
      do {
	// get the 'dim_theta + 1' initial points
	MultiplePointsGenerator(data, initPoints, dim_theta + 1, num_gauss);
	
	// obtaining the function values at those points
	for( index_t i = 0 ; i < dim_theta + 1 ; i++ ) {
	  initVal[i] = l2computations::L2Error(data, num_gauss, initPoints[i]) ;
	}
		
	// The optimizer ...
	cvgd = l2computations::PolytopeOptimizer(initPoints, initVal, dim_theta, 
						 ftol, (l2computations::L2Error), &niters, 
						 data, num_gauss);
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
      InitialPointGenerator(init_theta, data, num_gauss);
      // The optimizer.......
      l2computations::QuasiNewtonOptimizer(init_theta, dim_theta,
					   grad_tol, &niters, &final_val,
					   (l2computations::L2ErrorWithGradients),
					   data, num_gauss);
      
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

void MoGL2E::MultiplePointsGenerator(Matrix& d, double **points, 
				     index_t number_of_points,
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

void MoGL2E::InitialPointGenerator (double *theta, Matrix& data, 
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
