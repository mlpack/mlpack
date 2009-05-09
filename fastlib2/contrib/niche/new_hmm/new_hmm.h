#ifndef NEW_HMM_H
#define NEW_HMM_H

#include "fastlib/fastlib.h"
//#include<stdlib.h>
//#include<list>
//#include<math.h>

//using namespace std;


#define K 10
#define MAX_ITERATIONS 20
#define PERTURB 0.2 // 0.000002
#define INITIAL_N 2

#define ERGODIC 0
#define LEFT_RIGHT 1

#define HMM_TYPE LEFT_RIGHT


const fx_entry_doc new_hmm_entries_doc[] = {
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc new_hmm_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc new_hmm_doc = {
  new_hmm_entries_doc, new_hmm_submodules,
  "New HMM implementation\n"
};




typedef struct hard_cluster {
  Vector mean;
  list<int> point_indices;
  int num_points;
} hard_cluster;

/*
typedef struct soft_cluster {
  Vector mean;
  Vector probs;
  double sum_probs;
} soft_cluster;
*/

typedef struct distribution {
  Vector mu;
  Vector sigma; //lets stick with diagonal covariance matrices
} distribution;


typedef struct dataset {
  int T;
  Matrix b;

  Matrix alpha;
  Matrix beta;
  Matrix xi_self;
  Matrix xi_sum_over_t;
  Matrix gamma;
  Vector gamma_sum_over_t;
  Vector gamma_t_sum_over_i;
  Vector c; //scaling factors - incredibly important to avoid underflow!
} dataset;


typedef struct HMM {
  Vector pi;
  Matrix a;
  ArrayList<distribution> state_distributions;
  dataset datasets[K];
} HMM;



int N = INITIAL_N;
int D;
double pi_term = pow(2 * M_PI, ((double)D) / -2);
double ln_sum_T;






double det_diagonal(Vector v) {
  double product = 1;
  int length = v.length();
  int i;

  for(i = 0; i < length; i++) {
    product *= v[i];
  }
  
  return product;
}


double vector_sum(Vector v) {
  double sum = 0;
  int length = v.length();

  for(int i = 0; i < length; i++) {
    sum += v[i];
  }
  
  return sum;
}



// this function assumes a and b are of the same dimension
// behavior is unexpected otherwise!
Matrix dot_multiply(Matrix a, Matrix b) {
  int a_cols = a.n_cols();
  int a_rows = a.n_rows();

  int i, j;

  Matrix new_matrix;
  new_matrix.Init(a_cols, a_rows);

  for(i = 0; i < a_cols; i++) {
    for(j = 0; j < a_rows; j++) {
      new_matrix.set(i, j, a.get(i,j) * b.get(i,j));
    }
  }

  return new_matrix;
}

// this function assumes a and b are of the same dimension
// behavior is unexpected otherwise!
void dot_multiply_overwrite(Matrix *p_a, Matrix b) {

  Matrix &a = *p_a;


  int a_cols = a.n_cols();
  int a_rows = a.n_rows();

  int i, j;

  for(i = 0; i < a_cols; i++) {
    for(j = 0; j < a_rows; j++) {
      a.set(i, j, a.get(i,j) * b.get(i,j));
    }
  }
}


// this function assumes a and b are of the same dimension
// behavior is unexpected otherwise!
Matrix dot_divide(Matrix a, Matrix b) {
  int a_cols = a.n_cols();
  int a_rows = a.n_rows();

  int i, j;

  Matrix new_matrix;
  new_matrix.Init(a_cols, a_rows);

  for(i = 0; i < a_cols; i++) {
    for(j = 0; j < a_rows; j++) {
      double a_value = a.get(i,j);
      double b_value = b.get(i,j);

      new_matrix.set(i, j, a_value / b_value);
    }
  }

  return new_matrix;
}


// this function assumes a and b are of the same dimension
// behavior is unexpected otherwise!
Vector dot_multiply(Vector a, Vector b) {
  int length = a.length();

  int i;

  Vector new_vector;
  new_vector.Init(length);

  for(i = 0; i < length; i++) {
    new_vector[i] = a[i] * b[i];
  }

  return new_vector;
}

// this function assumes a and b are of the same dimension
// behavior is unexpected otherwise!
void dot_multiply_overwrite(Vector a, Vector b, Vector *p_v) {
  
  Vector &v = *p_v;
  
  int length = a.length();

  int i;

  for(i = 0; i < length; i++) {
    v[i] = a[i] * b[i];
  }

}


Vector dot_divide(Vector a, Vector b) {
  int length = a.length();

  int i;

  Vector new_vector;
  new_vector.Init(length);

  for(i = 0; i < length; i++) {
    new_vector[i] = a[i] / b[i];
  }

  return new_vector;
}


void display_matrix_no_label(Matrix a) {
  int i, j;

  index_t num_cols = a.n_cols();
  index_t num_rows = a.n_rows();

  printf("%d x %d\n", num_rows, num_cols);

  for(i = 0; i < num_rows; i++) {
    for(j = 0; j < num_cols; j++) {
      printf("%f ", a.get(i, j));
    }
    printf("\n");
  }

  printf("\n");
}



void display_matrix(const char *label, Matrix a) {

  printf("%s:\n", label);
  display_matrix_no_label(a);
}

void display_vector_no_label(Vector a) {

  int length = a.length();
  
  for(int i = 0; i < length; i++) {
    printf("%f ", a[i]);
  }

  printf("\n\n");
}


void display_vector(const char *label, Vector a) {
  printf("%s:\n", label);
  display_vector_no_label(a);
}




void display_state_distributions(ArrayList<distribution> state_distributions) {
  int i;

  for(i = 0; i < N; i++) {
    printf("\nstate %d\n", i);
    printf("mu:\n\t");
    display_vector_no_label(state_distributions[i].mu);
    printf("sigma:\n\t");
    display_vector_no_label(state_distributions[i].sigma);
  }
}





ArrayList<distribution> init_state_distributions(ArrayList<distribution> state_distributions, Matrix data) {
  int t, i, j, d;

  
  int T = data.n_cols();

  // k-means cluster the data

  ArrayList<hard_cluster> old_clusters;
  ArrayList<hard_cluster> new_clusters;

  old_clusters.Init(N);
  new_clusters.Init(N);

  //printf("initialize clusters\n");

  //initialize clusters
  for(i = 0; i < N; i++) {
    new_clusters[i].mean.Init(D);
    old_clusters[i].mean.Init(D);
    new_clusters[i].mean.SetZero();
    new_clusters[i].num_points = 0;
    new_clusters[i].point_indices.clear();
  }
  

  int data_indices[T];

  for(t = 0; t < T; t++) {
    data_indices[t] = t;
  }

  int deck_size = T;

  srand(time(NULL));


  int cluster_index = 0;
  while(deck_size > 0) {
    int draw = rand() % deck_size;

    Vector selected_point;
    data.MakeColumnVector(data_indices[draw], &selected_point);
    la::AddTo(selected_point, &(new_clusters[cluster_index].mean));
    new_clusters[cluster_index].num_points++;
    new_clusters[cluster_index].point_indices.push_back(data_indices[draw]);

    deck_size--;
    cluster_index = (cluster_index + 1) % N;

    int temp = data_indices[deck_size];
    data_indices[deck_size] = data_indices[draw];
    data_indices[draw] = temp;
  }



  for(i = 0; i < N; i++) {
    if(new_clusters[i].num_points > 0) {
      la::Scale(1 / ((double)(new_clusters[i].num_points)),
		&(new_clusters[i].mean));
    }
  }
  

  // a simple vector that lets us track cluster mean changes between iterations
  // important for stopping criteria!
  Vector change;
  change.Init(D);

  bool anyChanges;

  int num_iterations = 0;
  
  //start iterating
  do {

    // switch!
    ArrayList<hard_cluster> temp = old_clusters;
    old_clusters = new_clusters;
    new_clusters = temp;


  
    for(i = 0; i < N; i++) {
      new_clusters[i].mean.SetZero();
      new_clusters[i].num_points = 0;
      new_clusters[i].point_indices.clear();
    }
    
    
    for(int t = 0; t < T; t++) {
      Vector current_point;
      data.MakeColumnVector(t, &current_point);

      double min_dist = DBL_MAX;
      int best_cluster = 0;

      for(i = 0; i < N; i++) {
	double dist =
	  la::DistanceSqEuclidean(current_point, old_clusters[i].mean);

	if(dist < min_dist) {
	  min_dist = dist;
	  best_cluster = i;
	}
      }

      la::AddTo(current_point, &(new_clusters[best_cluster].mean));
      new_clusters[best_cluster].num_points++;
      new_clusters[best_cluster].point_indices.push_back(t);
    }

    for(i = 0; i < N; i++) {
      if(new_clusters[i].num_points > 0) {
	la::Scale(1 / ((double)(new_clusters[i].num_points)),
		  &(new_clusters[i].mean));
      }
    }



    anyChanges = false;
    
    for(i = 0; i < N; i++) {
      la::SubOverwrite(old_clusters[i].mean, new_clusters[i].mean, &change);
      for(j = 0; j < D; j++) {
	if(change[j] != 0) {
	  anyChanges = true;
	}
      }
    }

    num_iterations++;

  } while(anyChanges == true);

  //end iterating

  printf("clusters converged after %d iterations\n", num_iterations);

  /*
    printf("\nconverged cluster means\n");
    for(i = 0; i < N; i++) {
    for(j = 0; j < D; j++) {
    printf("%f ", new_clusters[i].mean[j]);
    }
    printf("\n");
    }
  */

  Vector result;
  result.Init(D);


  for(i = 0; i < N; i++) {
    state_distributions[i].mu.CopyValues(new_clusters[i].mean);
    state_distributions[i].sigma.SetZero();


    list<int>::iterator points_iterator;
    list<int> point_indices = new_clusters[i].point_indices;
    for(points_iterator = point_indices.begin(); points_iterator != point_indices.end(); points_iterator++) {

      Vector current_point;
      data.MakeColumnVector(*points_iterator , &current_point);

      la::SubOverwrite(new_clusters[i].mean, current_point, &result);

      for(d = 0; d < D; d++) {
	result[d] *= result[d];
      }

      la::AddTo(result, &(state_distributions[i].sigma));

    }

    if(new_clusters[i].num_points > 0) {
      la::Scale(1 / ((double)(new_clusters[i].num_points - 1)),
		&(state_distributions[i].sigma));
    }

  }


  //display_state_distributions(state_distributions);


  srand48(time(NULL));

  bool problems = true;

  while(problems) {
    problems = false;
    for(i = 0; i < N; i++) {
      bool empty = false;

      for(d = 0; d < D; d++) {
	if(isnan(state_distributions[i].sigma[d])) {
	  empty = true;
	}
	else if(state_distributions[i].sigma[d] < .00001) {
	  empty = true;
	}
      }

      if(empty) {

	j = rand() % N;
	
	state_distributions[i].mu.CopyValues(state_distributions[j].mu);
	state_distributions[i].sigma.CopyValues(state_distributions[j].sigma);

	// perturb each dimension proportional to variance of dimension
	for(d = 0; d < D; d++) {
	  state_distributions[i].mu[d] += 
	    (PERTURB * (0.5 -  drand48()) * state_distributions[i].sigma[d]);
	}
      }

      if(empty) {
	problems = true;
      }
    }
  }
  


  //display_state_distributions(state_distributions);
  
  return state_distributions;
  
}




ArrayList<distribution> init_left_right_state_distributions(ArrayList<distribution> state_distributions, Matrix all_data[]) {


  int t, i, d, k;
  
  for(i = 0; i < N; i++) {
    state_distributions[i].mu.SetZero();
    state_distributions[i].sigma.SetZero();
  }


  i = 0;

  Vector result;
  result.Init(D);

  int state_point_counts[N];

  for(i = 0; i < N; i++) {
    state_point_counts[i] = 0;
  }


  for(k = 0; k < K; k++) {
    int T = all_data[k].n_cols();
    Matrix data;
    data.Alias(all_data[k]);

    for(t = 0; t < T; t++) {
      
      Vector o_t;
      data.MakeColumnVector(t, &o_t);
      
      i = (int)(t * ((double)N / T));
      
      state_point_counts[i]++;
      
      la::AddTo(o_t, &(state_distributions[i].mu));
      
      result.CopyValues(o_t);
      for(d = 0; d < D; d++) {
	result[d] *= result[d];
      }
      
      la::AddTo(result, &(state_distributions[i].sigma));
      
    }
  }

  for(i = 0; i < N; i++) {
    la::Scale(1 / ((double)state_point_counts[i]),
	      &(state_distributions[i].mu));

    la::Scale(1 / ((double)state_point_counts[i]),
	      &(state_distributions[i].sigma));

    la::SubFrom(state_distributions[i].mu, &(state_distributions[i].sigma));

    /*printf("mu[%d]\n", i);
      display_vector_no_label(state_distributions[i].mu);

    printf("sigma[%d]\n", i);
    display_vector_no_label(state_distributions[i].sigma);*/
  }

  /*for(i = 0; i < N; i++) {
    printf("num[%d] = %d\n", i,state_point_counts[i]);
    }*/


  return state_distributions;

}




void init_left_right_pi(Vector *p_pi) {
  Vector &pi = *p_pi;

  pi.SetZero();
  pi[0] = 1;
}


void init_left_right_a(Matrix *p_a) {
  Matrix &a = *p_a;

  int i, j;

  /* initialize a for a left-to-right hmm  
     state 0 can go to states 0,1,2
     state 1 can go to states 1,2,3
     state k can go to states k,k+1,k+2
     state n-3 can go to states n-3,n-2,n-1
     state n-2 can go to states n-2,n-1
     state n-1 can go to state n-1
  */

  a.SetZero();

  for(i = 0; i < N - 2; i++) {
    for(j = 0; j < 3; j++) {
      a.set(i, i + j, ((double)1) / 3);
    }
  }

  if(N >= 2) {
    a.set(N - 2, N - 2, ((double)1) / 2);
    a.set(N - 2, N - 1, ((double)1) / 2);
  }
  
  if(N >= 1) {
    a.set(N - 1, N - 1, 1);
  }
}






double p_o_given_state_i(Vector x, distribution distr) {

  int i;

  Vector x_minus_mu;
  la::SubInit(distr.mu, x, &x_minus_mu);
  
  double det = 1;
  for(i = 0; i < D; i++) {
    det *= distr.sigma[i];
  }


  // dot divide rather than dot multiply in order
  // to dot multiply x_minus_mu and inverse(distr.sigma)
  double exponent =
    la::Dot(dot_divide(x_minus_mu, distr.sigma),
	    x_minus_mu);
  
  return (pi_term / sqrt(det)) * exp(-.5 * exponent);

}


void forward_recursion(Matrix *p_alpha, Vector *p_c,
			 Vector pi, Matrix a, Matrix b) {
 
  int T = b.n_cols();
 
  Matrix &alpha = *p_alpha;
  Vector &c = *p_c;
  
  // set alpha(t = 0)

  Vector alpha_t, alpha_t_plus_1;
  alpha.MakeColumnVector(0, &alpha_t);
  alpha_t_plus_1.Init(N);

  Vector b_0;
  b.MakeColumnVector(0, &b_0);
  alpha_t.CopyValues(dot_multiply(pi, b_0));

  
  double sum = vector_sum(alpha_t);

  c[0] = 1 / sum;

  la::Scale(c[0], &alpha_t);



  // alpha recursion for t = 1 .. T

  
  for(int t = 0; t < T - 1; t++) {
    alpha_t.Destruct();
    alpha_t_plus_1.Destruct();

    alpha.MakeColumnVector(t, &alpha_t);
    alpha.MakeColumnVector(t + 1, &alpha_t_plus_1);

    Vector b_t_plus_1;
    b.MakeColumnVector(t + 1, &b_t_plus_1);
    
    la::MulOverwrite(alpha_t, a, &alpha_t_plus_1);

    alpha_t_plus_1.CopyValues(dot_multiply(alpha_t_plus_1, b_t_plus_1));

    sum = vector_sum(alpha_t_plus_1);

    c[t + 1] = 1 / sum;

    la::Scale(c[t + 1], &alpha_t_plus_1);

  }

  //  return alpha;
}

void backward_recursion(Matrix *p_beta, Vector c,
			  Vector pi, Matrix a, Matrix b) {
  int T = b.n_cols();

  Matrix &beta = *p_beta;

  // set beta for t = T
  
  Vector beta_t, beta_t_plus_1;
  beta.MakeColumnVector(T - 1, &beta_t);
  beta_t_plus_1.Init(N);
  
  beta_t.SetAll(c[T - 1]);


  // beta recursion for t = T - 1 ... 0

  for(int t = T - 2; t >= 0; t--) {
    beta_t.Destruct();
    beta_t_plus_1.Destruct();

    beta.MakeColumnVector(t, &beta_t);
    beta.MakeColumnVector(t + 1, &beta_t_plus_1);

    Vector b_t_plus_1;
    b.MakeColumnVector(t + 1, &b_t_plus_1);

    beta_t.CopyValues(dot_multiply(b_t_plus_1, beta_t_plus_1));
    Vector result;
    la::MulInit(a, beta_t, &result);
    beta_t.CopyValues(result);

    la::Scale(c[t], &beta_t);

  }

  //return beta;
}




// returns probability of the optimal path through the HMM [pi, a, b]

void viterbi(Vector pi, Matrix a, Matrix b,
	     int q_star[], double *p_p_q_star) {

  int T = b.n_cols();
  
  double &p_q_star = *p_p_q_star;

  int t, i, j;

  Vector b_t;
  b.MakeColumnVector(0, &b_t);
  
  Vector delta_t_minus_1;
  delta_t_minus_1.Init(N);

  int psi[N][T];

  //set initial conditions

  for(i = 0; i < N; i++) {
    psi[i][0] = 0;
  }


  for(i = 0; i  < N; i++) {
    delta_t_minus_1[i] = log(pi[i]) + log(b_t[i]);
  }
  //display_vector("delta_t_minus_1", delta_t_minus_1);


  // recurse
  
  for(t = 1; t < T; t++) {
    for(j = 0; j < N; j++) {

      double max = -INFINITY;
      int argmax = -1;
      for(i = 0; i < N; i++) {
	double log_product = delta_t_minus_1[i] + log(a.get(i,j));
	if(log_product > max) {
	  max = log_product;
	  argmax = i;
	}
      }

      delta_t_minus_1[j] = max + log(b.get(j, t));
      psi[j][t] = argmax;
    }
  }


  
  // backtrack to calculate the Viterbi path
  
  double max = -INFINITY;
  int argmax = -1;
  for(i = 0; i < N; i++) {
    if(delta_t_minus_1[i] > max) {
      max = delta_t_minus_1[i];
      argmax = i;
    }
  }

  p_q_star = max;
  q_star[T-1] = argmax;

  for(t = T - 2; t >= 0; t--) {
    q_star[t] = psi[q_star[t + 1]][t + 1];
  }

}
  


double calculate_log_P_O_given_lambda(Vector c) {
  
  int T = c.length();

  double log_P_O_given_lambda = 0;
  
  for(int t = 0; t < T; t++) {
    log_P_O_given_lambda -= log(c[t]);
  }

  return log_P_O_given_lambda;
}



void update_b(Matrix *p_b, Matrix data,
	      ArrayList<distribution> state_distributions) {
  int T = data.n_cols();
  Matrix &b = *p_b;
  int t, i;

  for(t = 0; t < T; t++) {
    Vector o_t;
    data.MakeColumnVector(t, &o_t);
    
    for(i = 0; i < N; i++) {
      b.set(i, t, p_o_given_state_i(o_t, state_distributions[i]));
    }
  }
}


void update_pi(Vector *p_pi, dataset datasets[]) {
  Vector &pi = *p_pi;
  double pi_sum_over_i;
  int i, k;

  // accumulate
  for(i = 0; i < N; i++) {
    pi[i] = 0;
    for(k = 0; k < K; k++) {
      pi[i] += datasets[k].gamma.get(i, 0);
    }
  }
  
  //calculate normalization factor
  pi_sum_over_i = 0;
  for(i = 0; i < N; i++) {
    pi_sum_over_i += pi[i];
  }
  
  //normalize
  for(i = 0; i < N; i++) {
    pi[i] /= pi_sum_over_i;
  }
}



void update_a(Matrix *p_a, dataset datasets[]) {
  Matrix &a = *p_a;
  double xi_i_j_sum_over_t_k;
  double gamma_i_sum_over_t_k;
  int i, j, k;

  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      
      xi_i_j_sum_over_t_k = 0;
      gamma_i_sum_over_t_k = 0;
      
      for(k = 0; k < K; k++) {
	xi_i_j_sum_over_t_k += datasets[k].xi_sum_over_t.get(i, j);
	gamma_i_sum_over_t_k += datasets[k].gamma_sum_over_t[i];
      }

      if(gamma_i_sum_over_t_k > 0) {
	a.set(i, j, xi_i_j_sum_over_t_k / gamma_i_sum_over_t_k);
      }
    }
  }
}



void update_state_distributions(ArrayList<distribution> *p_state_distributions,
				dataset datasets[], Matrix all_data[]) {
  ArrayList<distribution> &state_distributions = *p_state_distributions;
  Vector scaled_o_t, result;
  int i, k, t, d;
  
  scaled_o_t.Init(D);
  result.Init(D);
  
  
  for(i = 0; i < N; i++) {
    //update mu's
    
    Vector mu;
    mu.Alias(state_distributions[i].mu);
    mu.SetZero();
    
    double gamma_sum_over_t_k = 0;
    
    for(k = 0; k < K; k++) {
      int T = datasets[k].T;
      Matrix data, gamma;
      data.Alias(all_data[k]);
      gamma.Alias(datasets[k].gamma);
      
      for(t = 0; t < T; t++) {
	
	Vector o_t;
	data.MakeColumnVector(t, &o_t);
	
	la::ScaleOverwrite(gamma.get(i, t), o_t, &scaled_o_t);
	
	la::AddTo(scaled_o_t, &mu);
      }
      gamma_sum_over_t_k += datasets[k].gamma_sum_over_t[i];
    }
    
    if(gamma_sum_over_t_k > 0) {
      la::Scale(1 / gamma_sum_over_t_k, &mu); //normalize mu
    }
    
    
    
    //update sigma's
    
    Vector sigma;
    sigma.Alias(state_distributions[i].sigma);
    sigma.SetZero();
      
    for(k = 0; k < K; k++) {
      int T = datasets[k].T;
      Matrix data, gamma;
      data.Alias(all_data[k]);
      gamma.Alias(datasets[k].gamma);
	
      for(t = 0; t < T; t++) {
	  
	Vector o_t;
	data.MakeColumnVector(t, &o_t);
	  
	la::SubOverwrite(mu, o_t, &result);
	  
	for(int d = 0; d < D; d++) {
	  result[d] *= result[d];
	}

	la::Scale(gamma.get(i, t), &result);
	  
	la::AddTo(result, &sigma);
      }
    }
      
    if(gamma_sum_over_t_k > 0) {
      la::Scale(1 / gamma_sum_over_t_k, &sigma); //normalize sigma
    }


    for(d = 0; d < D; d++) {
      if(sigma[d] < .01) {
	sigma[d] = .01;
	//printf("train_hmm sigma fixed!\n");
      }
    }
    
  }
}






void calculate_xi_sum_over_t(Matrix *p_xi_sum_over_t, Matrix *p_xi_self,
			     Matrix alpha, Matrix beta, Matrix a, Matrix b) {
  int T = b.n_cols();
  Matrix &xi_sum_over_t = *p_xi_sum_over_t;
  Matrix &xi_self = *p_xi_self;
  int t, j, i;

  xi_sum_over_t.SetZero();

      
  for(t = 0; t < T - 1; t++) {

    Matrix alpha_t;
    alpha.MakeColumnSlice(t, 1, &alpha_t);
	
    Vector beta_t_plus_1;
    beta.MakeColumnVector(t + 1, &beta_t_plus_1);
	
    Vector b_t_plus_1;
    b.MakeColumnVector(t + 1, &b_t_plus_1);
	
    Vector beta_b_vector = dot_multiply(beta_t_plus_1, b_t_plus_1);

    Matrix beta_b_matrix;
    beta_b_matrix.AliasRowVector(beta_b_vector);

    Matrix xi_t;
    la::MulInit(alpha_t, beta_b_matrix, &xi_t);

    dot_multiply_overwrite(&xi_t, a);

    double sum = 0;
    for(j = 0; j < N; j++) {
      for(i = 0; i < N; i++) {
	sum += xi_t.get(i,j);
      }
    }
  
    if(sum > 0) {
      la::Scale(1 / sum, &xi_t);
    }

    // we need these for use in the temporal split objective function
    for(i = 0; i < N; i++) {
      xi_self.set(i, t, xi_t.get(i, i));
    }

    la::AddTo(xi_t, &xi_sum_over_t);
    
    /*printf("\nxi_t for t = %d:\n", t);
      display_matrix_no_label(xi_t);*/ //massive
  }
}


void calculate_gamma(Matrix *p_gamma, Vector *p_gamma_sum_over_t,
		     Vector *p_gamma_t_sum_over_i_vector,
		     Matrix alpha, Matrix beta) {
  int T = alpha.n_cols();
  Matrix &gamma = *p_gamma;
  Vector &gamma_sum_over_t = *p_gamma_sum_over_t;
  Vector &gamma_t_sum_over_i_vector = *p_gamma_t_sum_over_i_vector;
  int t, i;

  gamma_sum_over_t.SetZero();
  
  for(t = 0; t < T; t++) {
    
    Vector gamma_t,alpha_t, beta_t;
    
    alpha.MakeColumnVector(t, &alpha_t);
    beta.MakeColumnVector(t, &beta_t);
    gamma.MakeColumnVector(t, &gamma_t);

    double gamma_t_sum_over_i = 0;
    for(i = 0; i < N; i++) {
      double product = alpha_t[i] * beta_t[i];
	  
      gamma_t[i] = product;
	  
      gamma_t_sum_over_i += product;
    }

    gamma_t_sum_over_i_vector[t] = gamma_t_sum_over_i;

    la::Scale(1 / gamma_t_sum_over_i, &gamma_t);

    if(t < T - 1) {
      la::AddTo(gamma_t, &gamma_sum_over_t);
    }
  }
}



void set_phoney_test_values(Vector *p_pi, Matrix *p_a, Matrix *p_b) {
  Vector &pi = *p_pi;
  Matrix &a = *p_a;
  Matrix &b = *p_b;

  // set totally phoney test values
  
  pi[0] = .3;
  pi[1] = .3;
  pi[2] = .4;

  a.set(0,0,.7); a.set(0,1,.25); a.set(0,2,.05);
  a.set(1,0,.25); a.set(1,1,.6); a.set(1,2,.15);
  a.set(2,0,.1); a.set(2,1,.1); a.set(2,2,.8);

  b.set(0,0,.9); b.set(0,1,.85); b.set(0,2,.2); b.set(0,3,.1); b.set(0,4,.1); 
  b.set(1,0,.2); b.set(1,1,.1); b.set(1,2,.9); b.set(1,3,.8); b.set(1,4,.05); 
  b.set(2,0,.1); b.set(2,1,.15); b.set(2,2,.2); b.set(2,3,.1); b.set(2,4,.95); 
}
  


void display_viterbi_path(Vector pi, Matrix a, Matrix b) {
  int T = b.n_cols();
  int q_star[T];
  double p_q_star;
  int t;
  
  viterbi(pi, a, b, q_star, &p_q_star);
  
  
  printf("p(q*) = %f\n", p_q_star);
  printf("q*:\n");
  for(t = 0; t < T; t++) {
    printf("%d\n", q_star[t] + 1);
  }
  printf("\n\n");
}








void train_hmm(HMM *p_hmm, Matrix all_data[]) {

  HMM &hmm = *p_hmm;
  
  Vector pi; pi.Alias(hmm.pi);
  Matrix a; a.Alias(hmm.a);
  
  ArrayList<distribution> &state_distributions = hmm.state_distributions;
  dataset *datasets = hmm.datasets;

  int k;

//   display_vector("pi", pi);
//   display_matrix("a", a);
//   display_state_distributions(state_distributions);
  
  

  int num_iterations = 0;

  do {
    num_iterations++;
    //printf("iteration %d\n\n", num_iterations);
        
    for(k = 0; k < K; k++) {
    
      Matrix b, xi_sum_over_t, xi_self, gamma;
      Vector gamma_sum_over_t, gamma_t_sum_over_i, c;
      
      b.Alias(datasets[k].b);
      xi_sum_over_t.Alias(datasets[k].xi_sum_over_t);
      xi_self.Alias(datasets[k].xi_self);
      gamma.Alias(datasets[k].gamma);
      gamma_sum_over_t.Alias(datasets[k].gamma_sum_over_t);
      gamma_t_sum_over_i.Alias(datasets[k].gamma_t_sum_over_i);
      c.Alias(datasets[k].c);


      Matrix alpha;
      alpha.Alias(datasets[k].alpha);
      /*datasets[k].alpha = */forward_recursion(&alpha, &c, pi, a, b);
//       display_matrix("alpha", alpha); //massive

      Matrix beta;
      beta.Alias(datasets[k].beta);
      /*datasets[k].beta = */backward_recursion(&beta, c, pi, a, b);
//       display_matrix("beta", beta); //massive
  
      //calculate xi accumulated over t
      calculate_xi_sum_over_t(&xi_sum_over_t, &xi_self, alpha, beta, a, b);
      //display_matrix("xi_sum_over_t", xi_sum_over_t); //massive
      
      // calculate gamma and gamma accumulated over t
      calculate_gamma(&gamma, &gamma_sum_over_t,
		      &gamma_t_sum_over_i,
		      alpha, beta);
      /*display_matrix("gamma", gamma);
	display_vector("gamma_sum_over_t", gamma_sum_over_t); // massive */
    }
    
  

    // UPDATE PARAMETERS
  
    // update pi
    update_pi(&pi, datasets);
//     display_vector("pi", pi);
    
    // update a
    update_a(&a, datasets);
//     display_matrix("a", a);
  
    //update state_distributions
    update_state_distributions(&state_distributions, datasets, all_data);
//     display_state_distributions(state_distributions);

    

    // update b = P(x | S_i) using updated state distributions
    for(k = 0; k < K; k++) {
      update_b(&(datasets[k].b), all_data[k], state_distributions);
      //display_matrix("b", datasets[k].b); //massive
    }

    /*
    for(k = 0; k < K; k++) {
      printf("c[%d]\n", k);
      display_vector_no_label(datasets[k].c);
      printf("alpha[%d]\n", k);
      display_matrix_no_label(datasets[k].alpha);
      printf("beta[%d]\n", k);
      display_matrix_no_label(datasets[k].beta);
      printf("gamma[%d]\n", k);
      display_matrix_no_label(datasets[k].gamma);
    }

    printf("ITERATE\n");
    */
    
    //display_state_distributions(state_distributions);
    
  } while(num_iterations < MAX_ITERATIONS);


  //display_state_distributions(state_distributions);


}



/*
double test_hmm(HMM hmm, const char *test_filename) {

  Matrix data;
  data::Load(test_filename, &data);
  int T = data.n_cols();
  
  Matrix alpha;
  alpha.Init(N, T);

  Vector c;
  c.Init(T);
  
  Matrix b;
  b.Init(N, T);

  
  update_b(&b, data, hmm.state_distributions);
  
  forward_recursion(&alpha, &c, hmm.pi, hmm.a, b);
  
  return calculate_log_P_O_given_lambda(c);

}
*/




#endif /* NEW_HMM_H */
