#include "fastlib/fastlib.h"
#include "../fastica/lin_alg.h"

void SaveCorrectly(const char *filename, Matrix a) {
  Matrix a_transpose;
  la::TransposeInit(a, &a_transpose);
  data::Save(filename, a_transpose);
}


void RADICALOptTheta(Matrix X, double std_dev, index_t m,
		     index_t reps, double new_k, double range,
		     double *theta_star, Matrix* rotator_star) {
  Matrix X_aug;

  index_t d = X.n_rows();
  index_t n = X.n_cols();

  
  if(reps == 1) {
    X_aug.Copy(X);
  }
  else {
    Matrix noise_matrix;
    Scale(std_dev,
	  RandNormalInit(d, n * reps, &noise_matrix));
    
    X_aug.Init(d, n * reps);
    
    double* X_elements = X.ptr();
    double* X_aug_elements = X_aug.ptr();
    
    index_t num_elements = d * n;
    for(index_t rep_num = 0; rep_num < reps; rep_num++) {
      memcpy(X_aug_elements + (rep_num * num_elements) * sizeof(double),
	     X_elements,
	     num_elements * sizeof(double));
    }

    la::AddTo(noise_matrix, &X_aug);
  }
}



void RADICAL(Matrix X, Matrix whitening_matrix, Matrix X_whitened,
	     index_t k, bool aug_flag, index_t reps, double std_dev, index_t m,
	     Matrix* W, Matrix* Y) {

  index_t d = X.n_rows();
  index_t n = X.n_cols();
  
  
  Matrix X_t, X_current_t, X_whitened_t;
  la::TransposeInit(X, &X_t);
  la::TransposeInit(X_whitened, &X_current_t);
  la::TransposeInit(X_whitened, &X_whitened_t);
  

  index_t sweeps = d - 1;
  Matrix* old_total_rotator;
  Matrix* total_rotator;
  Matrix* temp_total_rotator;
  DiagMatrixInit(d, 1, old_total_rotator);
  DiagMatrixInit(d, 1, total_rotator);

  index_t sweep_iteration = 0;



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



    
    
  


	





int RADICALMain(datanode *module, Matrix *W, Matrix *Y) {

  const char *data = fx_param_str_req(NULL, "data");
  
  Matrix X, X_centered, X_whitened, whitening_matrix;
  data::Load(data, &X);
  
  index_t d = X.n_rows(); // number of dimensions
  index_t n = X.n_cols(); // number of points


  index_t k = 150;
  bool aug_flag = false;
  index_t reps = 30;
  double std_dev = 0.175;

  index_t m = (index_t) floor(sqrt(n));

  Center(X, &X_centered);
  WhitenUsingSVD(X_centered, &X_whitened, &whitening_matrix);


  RADICAL(X, whitening_matrix, X_whitened, k, aug_flag, reps, std_dev, m, W, Y);

	



  W -> Init(0,0);
  Y -> Init(0,0);

  return SUCCESS_PASS;
}



int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  srand48(time(0));

  Matrix W, Y;

  int ret_val = RADICALMain(fx_root, &W, &Y);

  //SaveCorrectly("unmixing_matrix.dat", W);
  //SaveCorrectly("indep_comps.dat", Y);

  fx_done();

  return ret_val;
}
