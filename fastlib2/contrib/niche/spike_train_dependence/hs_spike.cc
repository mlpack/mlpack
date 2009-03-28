#include "fastlib/fastlib.h"
#include "hs_spike.h"


void ComputeKernelMatrix(Matrix x, Matrix* kernel_matrix) {
  int n_points = x.n_cols();
  int n_dims = x.n_rows();

  kernel_matrix -> Init(n_points, n_points);

  Vector diff;
  diff.Init(n_dims);
  for(int i = 0; i < n_points; i++) {
    Vector x_i;
    x.MakeColumnVector(i, &x_i);
    for(int j = i; j < n_points; j++) {
      Vector x_j;
      x.MakeColumnVector(j, &x_j);
      la::SubOverwrite(x_i, x_j, &diff);
      double val = exp(-1 * la::Dot(diff, diff));
      kernel_matrix -> set(i, j, val);
      if(i != j) {
	kernel_matrix -> set(j, i, val);
      }
    }
  }
}


int main(int argc, char* argv[]) {
  
  fx_module* root = fx_init(argc, argv, NULL);
  
  const char* x_filename = fx_param_str(NULL, "x", "x.csv");
  const char* y_filename = fx_param_str(NULL, "y", "y.csv");

  const int dependence_horizon = fx_param_int(NULL, "tau", 2);

  SpikeSeqPair spikes_pair;
  spikes_pair.Init(x_filename, y_filename, dependence_horizon);

  //spikes_pair.Merge();
  //spikes_pair.PrintAllSpikes();

  Matrix x1, y1, x2, y2;
  spikes_pair.ConstructPoints(&x1, &y1, &x2, &y2);

  int n_points = x1.n_cols();

  Matrix K;
  ComputeKernelMatrix(x1, &K);

  Matrix L;
  ComputeKernelMatrix(y1, &L);

  Matrix H;
  H.Init(n_points, n_points);
  H.SetZero();
  for(int i = 0; i < n_points; i++) {
    H.set(i, i, 1);
  }
  double one_over_n_points = ((double)1) / ((double) n_points);
  for(int i = 0; i < n_points; i++) {
    for(int j = 0; j < n_points; j++) {
      H.set(j, i, H.get(j, i) - one_over_n_points);
    }
  }
  K.PrintDebug("K");
  L.PrintDebug("L");
  H.PrintDebug("H");

  Matrix result, result2;
  la::MulInit(H, K, &result);
  la::MulInit(result, H, &result2);
  la::MulOverwrite(result2, L, &result);
  la::Scale(((double)1) / ((double)(n_points * n_points)), &result);
  double hsic = la::Trace(result);
  printf("hsic = %e\n", hsic);
  /*
  y1.PrintDebug("y1");
  y1_kernel_matrix.PrintDebug("y1 kernel matrix");
  */
  /*
  y1.PrintDebug("y1");
  x2.PrintDebug("x2");
  y2.PrintDebug("y2");
  */


  


  
  

  fx_done(root);

}
