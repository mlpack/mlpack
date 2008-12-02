#include "fastlib/fastlib.h"

const fx_entry_doc kernel_matrix_generator_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   " File consists of data points in lines.\n"},
  {"bandwidth", FX_REQUIRED, FX_DOUBLE, NULL,
   " The kernel bandwidth.\n"},
  {"rate", FX_PARAM, FX_INT, NULL,
   " The sampling rate, default = 20.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   " File to hold output kernel matrix,"
   " default = \"kernel_matrix.txt\".\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc kernel_matrix_generator_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc kernel_matrix_generator_doc = {
  kernel_matrix_generator_entries, kernel_matrix_generator_submodules,
  "This is a program generating the kernel matrix of "
  "a set of points.\n"
};

int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  fx_init(argc, argv, &kernel_matrix_generator_doc);
  
  Matrix references;
  const char *references_file_name = fx_param_str_req(fx_root, "data");
  double bandwidth = fx_param_double_req(fx_root, "bandwidth");
  data::Load(references_file_name, &references);

  printf("nrows = %d ncols = %d", references.n_rows(), 
	 references.n_cols());
#define RUNNING 1
#if RUNNING==1  
  // Kernel matrix to be outputted.
  Matrix kernel_matrix;

  // Initialize the kernel.
  GaussianKernel kernel;
  kernel.Init(bandwidth);

  int rate = fx_param_int(fx_root, "rate", 20);
  int n_cols = (references.n_cols()-1)/rate+1;
  kernel_matrix.Init(n_cols, n_cols);
  for(index_t r = 0; r < references.n_cols(); r+=rate) {
    const double *r_col = references.GetColumnPtr(r);

    for(index_t q = 0; q < references.n_cols(); q+=rate) {
      const double *q_col = references.GetColumnPtr(q);
      double dsqd = la::DistanceSqEuclidean(references.n_rows(), q_col,
					    r_col);
      kernel_matrix.set(q/rate, r/rate, kernel.EvalUnnormOnSq(dsqd));
    }
  }

  // Output the matrix.
  const char *file_name = fx_param_str(fx_root, "output", 
				       "kernel_matrix.txt");
  data::Save(file_name, kernel_matrix);
  /*
  FILE *output_file = fopen(file_name, "w+");
  for(index_t r = 0; r < kernel_matrix.n_rows(); r++) {
    for(index_t c = 0; c < kernel_matrix.n_cols(); c++) {
      fprintf(output_file, "%g ", kernel_matrix.get(c, r));
    }
    fprintf(output_file, "\n");
  }
  */
  fx_done(fx_root);
#endif
  return 0;
}
