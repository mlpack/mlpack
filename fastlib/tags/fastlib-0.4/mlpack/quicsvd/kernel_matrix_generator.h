#include "fastlib/fastlib.h"

int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  fx_init(argc, argv, &kde_main_doc);
  
  Matrix references;
  const char *references_file_name = fx_param_str_req(fx_root, "data");
  double bandwidth = fx_param_double_req(fx_root, "bandwidth");
  data::Load(references_file_name, &references);
  
  // Kernel matrix to be outputted.
  Matrix kernel_matrix;

  // Initialize the kernel.
  GaussianKernel kernel;
  kernel.Init(bandwidth);

  kernel_matrix.Init(references.n_cols(), references.n_cols());
  for(index_t r = 0; r < references.n_cols(); r++) {
    const double *r_col = references.GetColumnPtr(r);

    for(index_t q = 0; q < references.n_cols(); q++) {
      
      double dsqd = la::DistanceSqEuclidean(references.n_rows(), q_col,
					    r_col);
      const double *q_col = references.GetColumnPtr(q);
      kernel_matrix.set(q, r, kernel.EvalUnnormOnSq(dsqd));
    }
  }

  // Output the matrix.
  const char *file_name = "kernel_matrix.txt";
  FILE *output_file = fopen(file_name, "w+");
  for(index_t r = 0; r < references.n_cols(); r++) {
    for(index_t c = 0; c < references.n_cols(); c++) {
      fprintf(output_file, "%g ", kernel_matrix.get(c, r));
    }
    fprintf(output_file, "\n");
  }

  fx_done(fx_root);
  return 0;
}
