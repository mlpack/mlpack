#include "fastlib/fastlib.h"
#include "../fastica/lin_alg.h"

void SaveCorrectly(const char *filename, Matrix a) {
  Matrix a_transpose;
  la::TransposeInit(a, &a_transpose);
  data::Save(filename, a_transpose);
}


int compare_doubles_ascending(const void* a, const void* b) {
  double* arg1 = (double*) a;
  double* arg2 = (double*) b;
  if(*arg1 < *arg2) {
    return -1;
  }
  else if(*arg1 == *arg2) {
    return 0;
  }
  else {
    return 1;
  }
}
  

double vasicekm(Vector v, index_t m) {

  Vector sorted_v;

  index_t n = v.length();

  sorted_v.Copy(v);

  qsort(sorted_v.ptr(), n, sizeof(double), compare_doubles_ascending);

  double sum_logs = 0;

  for(index_t i = 0; i < n; i++) {
    if((i + m) > (n - 1)) {
      sum_logs = sum_logs + log(sorted_v[n - 1] - sorted_v[i - m]);
    }
    else if((i - m) < 0) {
      sum_logs = sum_logs + log(sorted_v[i + m] - sorted_v[0]);
    }
    else {
      sum_logs = sum_logs + log(sorted_v[i + m] - sorted_v[i - m]);
    }
  }

  double h = (sum_logs / n) + log(n / (2 * m));

  return h;

}



void RADICALOptTheta(Matrix X_t, double std_dev, index_t m,
		     index_t reps, double k, double range,
		     double *theta_star, Matrix* rotator_star) {
  Matrix X_aug_t;

  index_t d = X_t.n_cols();
  index_t n = X_t.n_rows();

  
  if(reps == 1) {
    X_aug_t.Alias(X_t);
  }
  else {
    Matrix noise_matrix;
    Scale(std_dev,
	  RandNormalInit(n * reps, d, &noise_matrix));
    
    X_aug_t.Init(n * reps, d);
    
    double* X_t_elements = X_t.ptr();
    double* X_aug_t_elements = X_aug_t.ptr();
    
    index_t num_elements = d * n;
    for(index_t rep_num = 0; rep_num < reps; rep_num++) {
      memcpy(X_aug_t_elements + (rep_num * num_elements) * sizeof(double),
	     X_t_elements,
	     num_elements * sizeof(double));
    }

    la::AddTo(noise_matrix, &X_aug_t);
  }

  Vector entropy;
  entropy.Init((index_t) k);

  for(index_t i = 0; i < k; i++) {
    double theta =
      ((double) (i - 1) / (k - 1) * M_PI / 2) - (M_PI / 4);

    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    Matrix rotator;
    rotator.Init(2,2);
    rotator.set(0,0, cos_theta);
    rotator.set(0,1, -sin_theta);
    rotator.set(1,0, sin_theta);
    rotator.set(1,1, cos_theta);
    Matrix rotated_X_aug_t;
    la::MulTransBInit(X_aug_t, rotator, &rotated_X_aug_t);

    Vector marginal_at_theta;
    marginal_at_theta.Init(d);
    for(index_t j = 0; j < d; j++) {
      Vector col_vector;
      rotated_X_aug_t.MakeColumnVector(j, &col_vector);
      marginal_at_theta[j] = vasicekm(col_vector, m);
    }

    entropy[i] = Sum(&marginal_at_theta);
  }

  double min_entropy = DBL_MAX;
  index_t arg_min_entropy = 0;

  for(index_t i = 0; i < k; i++) {
    if(entropy[i] < min_entropy) {
      arg_min_entropy = i;
      min_entropy = entropy[i];
    }
  }
    
  *theta_star = ((arg_min_entropy - 1) / (k - 1) * M_PI / 2) - (M_PI / 4);

  printf("rotated %5.2f degrees.\n", (*theta_star) / (2 * M_PI) * 360);

  rotator_star -> Init(2, 2);

  double cos_theta_star = cos(*theta_star);
  double sin_theta_star = sin(*theta_star);
  rotator_star -> set(0, 0, cos_theta_star);
  rotator_star -> set(0, 1, -sin_theta_star);
  rotator_star -> set(1, 0, sin_theta_star);
  rotator_star -> set(1, 1, cos_theta_star);
  
}



  void RADICAL(Matrix X, Matrix whitening_matrix, Matrix X_whitened,
	       index_t k, bool aug_flag, index_t reps, double std_dev, index_t m,
	       Matrix* W, Matrix* Y) { 

  index_t d = X.n_rows();
  //  index_t n = X.n_cols();
  
  
  Matrix X_t, X_current_t, X_whitened_t;
  la::TransposeInit(X, &X_t);
  la::TransposeInit(X_whitened, &X_current_t);
  X_whitened_t.Copy(X_current_t);
  

  index_t sweeps = d - 1;
  Matrix* old_total_rotator = NULL;
  Matrix* total_rotator = NULL;
  Matrix* temp_total_rotator = NULL;
  DiagMatrixInit(d, 1, old_total_rotator);
  DiagMatrixInit(d, 1, total_rotator);



  index_t final_k = k;
  double start_k_float = final_k / pow(1.3, ceil(sweeps / 2));
  double new_k_float = start_k_float;

  for(index_t sweep_num = 1; sweep_num < sweeps; sweep_num++) {
    printf("Sweep # %d of %d.\n", sweep_num, sweeps);
    double range = M_PI / 2;

    double new_k;
    if(sweep_num > ((double) sweeps / 2)) {
      new_k_float *= 1.3;
      new_k = floor(new_k_float);
    }
    else {
      new_k_float = start_k_float;
      new_k = max(30, (int) floor(new_k_float));
    }

    for(index_t i = 0; i < d - 1; i++) {
      for(index_t j = i + 1; j < d; j++) {
	Matrix cur_subspace_t;
	Vector selected_columns;
	selected_columns.Init(2);
	selected_columns[0] = i;
	selected_columns[1] = j;
	MakeSubMatrixByColumns(selected_columns, X_current_t, &cur_subspace_t);

	double theta_star;
	Matrix rotator_star;
	RADICALOptTheta(cur_subspace_t, std_dev, m, reps, new_k, range,
			&theta_star, &rotator_star);

	Matrix new_rotator;
	DiagMatrixInit(d, 1, &new_rotator);

	new_rotator.set(i, i, cos(theta_star));
	new_rotator.set(i, j, -sin(theta_star));
	new_rotator.set(j, i, sin(theta_star));
	new_rotator.set(j, j, cos(theta_star));
	
	temp_total_rotator = old_total_rotator;
	old_total_rotator = total_rotator;
	total_rotator = temp_total_rotator;
	la::MulInit(new_rotator, *old_total_rotator, total_rotator);
	  
	la::MulTransAOverwrite(X_whitened_t, *total_rotator, &X_current_t);
      }
    }
  }


  la::MulInit(*total_rotator, whitening_matrix, W);
  la::MulInit(*W, X, Y);
}



    
    
  


	



// there should be some way to wrap X, W, and Y into our module no? ask ryan

int RADICALMain(datanode *module, Matrix X, Matrix *W, Matrix *Y) {

  Matrix X_centered, X_whitened, whitening_matrix;

  index_t n = X.n_cols();
  
  index_t k = 150;
  bool aug_flag = false;
  index_t reps = 30;
  double std_dev = 0.175;

  index_t m = (index_t) floor(sqrt(n));

  Center(X, &X_centered);
  WhitenUsingSVD(X_centered, &X_whitened, &whitening_matrix);


  RADICAL(X, whitening_matrix, X_whitened,
	  k, aug_flag, reps, std_dev, m,
	  W, Y);



  return SUCCESS_PASS;
}



int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  Matrix X, W, Y;

  const char *data = fx_param_str_req(NULL, "data");
  data::Load(data, &X);
  
  int ret_val = RADICALMain(fx_root, X, &W, &Y);

  //SaveCorrectly("unmixing_matrix.dat", W);
  //SaveCorrectly("indep_comps.dat", Y);

  fx_done();

  return ret_val;
}
