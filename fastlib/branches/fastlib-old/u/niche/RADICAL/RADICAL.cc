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
  

double VasicekEntropyEstimate(Vector v, index_t m) {

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



void RADICALOptTheta(Matrix x_t_mat, double std_dev, index_t m,
		     index_t reps, index_t k, double range,
		     double *theta_star, Matrix* rotator_star_mat) {
  Matrix x_aug_t_mat;

  index_t d = x_t_mat.n_cols();
  index_t n = x_t_mat.n_rows();

  
  if(reps == 1) {
    x_aug_t_mat.Alias(x_t_mat);
  }
  else {
    RepeatMatrix(reps, 1, x_t_mat, &x_aug_t_mat);
    
    Matrix noise_mat;
    RandNormalInit(n * reps, d, &noise_mat);
    
    la::AddExpert(std_dev, noise_mat, &x_aug_t_mat);
  }

  Vector entropy;
  entropy.Init(k);

  for(index_t i = 0; i < k; i++) {
    double theta =
      ((double) (i - 1) / (double) (k - 1) * M_PI / 2) - (M_PI / 4);

    double cos_theta = cos(theta);
    double sin_theta = sin(theta);
    Matrix rotator_mat;
    rotator_mat.Init(2,2);
    rotator_mat.set(0,0, cos_theta);
    rotator_mat.set(0,1, -sin_theta);
    rotator_mat.set(1,0, sin_theta);
    rotator_mat.set(1,1, cos_theta);
    Matrix rotated_x_aug_t_mat;
    la::MulTransBInit(x_aug_t_mat, rotator_mat, &rotated_x_aug_t_mat);

    Vector marginal_at_theta;
    marginal_at_theta.Init(d);
    for(index_t j = 0; j < d; j++) {
      Vector col_vec;
      rotated_x_aug_t_mat.MakeColumnVector(j, &col_vec);
      marginal_at_theta[j] = VasicekEntropyEstimate(col_vec, m);
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
    
  *theta_star =
    ((double) (arg_min_entropy - 1) / (double) (k - 1) * M_PI / 2) - (M_PI / 4);

  printf("rotated %5.2f degrees\n", (*theta_star) / (2 * M_PI) * 360);

  rotator_star_mat -> Init(2, 2);

  double cos_theta_star = cos(*theta_star);
  double sin_theta_star = sin(*theta_star);
  rotator_star_mat -> set(0, 0, cos_theta_star);
  rotator_star_mat -> set(0, 1, -sin_theta_star);
  rotator_star_mat -> set(1, 0, sin_theta_star);
  rotator_star_mat -> set(1, 1, cos_theta_star);
  
}



  void RADICAL(Matrix x_mat, Matrix whitening_mat, Matrix x_whitened_mat,
	       index_t k, index_t reps, double std_dev, index_t m,
	       Matrix* w_mat, Matrix* y_mat) { 

  index_t d = x_mat.n_rows();
  //  index_t n = X.n_cols();
  
  
  Matrix x_t_mat, x_current_t_mat, x_whitened_t_mat;
  la::TransposeInit(x_mat, &x_t_mat);
  la::TransposeInit(x_whitened_mat, &x_current_t_mat);
  x_whitened_t_mat.Copy(x_current_t_mat);


  index_t sweeps = d - 1;
  Matrix* old_total_rotator_mat = new Matrix;
  Matrix* total_rotator_mat = new Matrix;
  Matrix* temp_total_rotator_mat;

  DiagMatrixInit(d, 1, old_total_rotator_mat); // set to identity matrix
  DiagMatrixInit(d, 1, total_rotator_mat); // set to identity matrix



  double start_k_float = (double) k / pow(1.3, ceil(sweeps / 2));
  double new_k_float = start_k_float;

  for(index_t sweep_num = 1; sweep_num < sweeps; sweep_num++) {
    printf("Sweep # %d of %d.\n", sweep_num, sweeps);
    double range = M_PI / 2;

    index_t new_k;
    if(sweep_num > ((double) sweeps / 2)) {
      new_k_float *= 1.3;
      new_k = (index_t) floor(new_k_float);
    }
    else {
      new_k_float = start_k_float;
      new_k = max(30, (index_t) floor(new_k_float));
    }

    for(index_t i = 0; i < d - 1; i++) {
      for(index_t j = i + 1; j < d; j++) {
	Matrix cur_subspace_t_mat;
	Vector selected_columns_vec;
	selected_columns_vec.Init(2);
	selected_columns_vec[0] = i;
	selected_columns_vec[1] = j;
	MakeSubMatrixByColumns(selected_columns_vec,
			       x_current_t_mat, &cur_subspace_t_mat);

	double theta_star;
	Matrix rotator_star_mat;
	RADICALOptTheta(cur_subspace_t_mat, std_dev, m, reps, new_k, range,
			&theta_star, &rotator_star_mat);

	Matrix new_rotator_mat;
	DiagMatrixInit(d, 1, &new_rotator_mat);

	double cos_theta_star = cos(theta_star);
	double sin_theta_star = sin(theta_star);
	
	new_rotator_mat.set(i, i, cos_theta_star);
	new_rotator_mat.set(i, j, -sin_theta_star);
	new_rotator_mat.set(j, i, sin_theta_star);
	new_rotator_mat.set(j, j, cos_theta_star);
	
	temp_total_rotator_mat = old_total_rotator_mat;
	old_total_rotator_mat = total_rotator_mat;
	total_rotator_mat = temp_total_rotator_mat;
	la::MulInit(new_rotator_mat,
		    *old_total_rotator_mat, 
		    total_rotator_mat);
	
	la::MulTransBOverwrite(x_whitened_t_mat,
			       *total_rotator_mat,
			       &x_current_t_mat);
      }
    }
  }


  la::MulInit(*total_rotator_mat, whitening_mat, w_mat);
  la::MulInit(*w_mat, x_mat, y_mat);
}


	

// there should be some way to wrap X, W, and Y into our module no? ask ryan

int RADICALMain(datanode* module, Matrix x_mat, Matrix* w_mat, Matrix* y_mat) {

  Matrix x_centered_mat, x_whitened_mat, whitening_mat;

  index_t n = x_mat.n_cols();
  index_t m = (index_t) floor(sqrt(n));
  
  index_t k = 150;
  index_t reps = 30;
  double std_dev = 0.175;



  Center(x_mat, &x_centered_mat);
  WhitenUsingSVD(x_centered_mat, &x_whitened_mat, &whitening_mat);

  RADICAL(x_mat, whitening_mat, x_whitened_mat,
	  k, reps, std_dev, m, w_mat, y_mat);



  return SUCCESS_PASS;
}



int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  Matrix x_mat, w_mat, y_mat;

  const char *data = fx_param_str_req(NULL, "data");
  data::Load(data, &x_mat);
  
  int ret_val = RADICALMain(fx_root, x_mat, &w_mat, &y_mat);

  //SaveCorrectly("unmixing_matrix.dat", W);
  //SaveCorrectly("indep_comps.dat", Y);

  fx_done();

  return ret_val;
}
