#include "fastlib/fastlib.h"
#include "../fastica/lin_alg.h"

void SaveCorrectly(const char *filename, Matrix a) {
  Matrix a_transpose;
  la::TransposeInit(a, &a_transpose);
  data::Save(filename, a_transpose);
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
  


  index_t sweeps = d - 1;
  Matrix old_total_rotator, total_rotator;
  DiagMatrixInit(d, 1, &old_total_rotator);
  DiagMatrixInit(d, 1, &total_rotator);

  index_t sweep_iteration = 0;


  Matrix X_current;
  X_current.Copy(X_whitened);


  index_t final_k = k;
  double start_k_float = final_k / pow(1.3, ceil(sweeps / 2));
  double new_k_float = start_k_float;

  for(index_t sweep_num = 1; sweep_num < sweeps; sweep_num++) {
    printf(1, "Sweep % %d of %d.\n", sweep_num, sweeps);
    double range = M_PI / 2;

    if(sweep_num > ((double) sweeps / 2)) {
      new_k_float *= 1.3;
      new_k = floor(new_k_float);





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
