#include "fastlib/fastlib.h"
#include <fastlib/fx/io.h>

PARAM_STRING_REQ("data", "File consists of data points in lines.", "kernel");
PARAM_STRING_REQ("kernel", "The kernel type: gaussian / polynomial", "kernel");
PARAM_STRING("output", "File to hold output kernel matrix default =\
 kernel_matrix.txt.", "kernel", "kernel_matrix.txt");
PARAM_INT("rate", "The sampling rate, default = 20 choose 1.", "kernel", 20);
PARAM(double, "bandwidth", "The gaussian kernel bandwidth, default = 1.",
  "kernel", 1, false);
PARAM(double, "degree", "The polynomial kernel degree, default = 1.",
  "kernel", 1, false);

PROGRAM_INFO("Kernel Generator", 
  "This is a program generating the kernel matrix of a set of points.");

using namespace mlpack;

void gen_gaussian_kernel(const Matrix& ref, Matrix* kernel_matrix,
			 int rate) {
  double bandwidth = IO::GetParam<double>("kernel/bandwidth");

  // Initialize the kernel.
  GaussianKernel kernel;
  kernel.Init(bandwidth);

  int n_cols = (ref.n_cols()-1)/rate+1;
  kernel_matrix->Init(n_cols, n_cols);
  for(index_t r = 0; r < ref.n_cols(); r+=rate) {
    const double *r_col = ref.GetColumnPtr(r);
    
    for(index_t q = 0; q < ref.n_cols(); q+=rate) {
      const double *q_col = ref.GetColumnPtr(q);
      double dsqd = la::DistanceSqEuclidean(ref.n_rows(), q_col,
					    r_col);
      kernel_matrix->set(q/rate, r/rate, kernel.EvalUnnormOnSq(dsqd));
    }
  }
}

void gen_polynomial_kernel(const Matrix& ref, Matrix* kernel_matrix,
			   int rate) {
  double degree = IO::GetParam<double>("kernel/degree");

  int n_cols = (ref.n_cols()-1)/rate+1;
  kernel_matrix->Init(n_cols, n_cols);
  for(index_t r = 0; r < ref.n_cols(); r+=rate) {
    const double *r_col = ref.GetColumnPtr(r);
    
    for(index_t q = 0; q < ref.n_cols(); q+=rate) {
      const double *q_col = ref.GetColumnPtr(q);
      double dsqd = la::Dot(ref.n_rows(), q_col, r_col) + 1;
      kernel_matrix->set(q/rate, r/rate, pow(dsqd, degree));
    }
  }
}

int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  fx_init(argc, argv, &kernel_matrix_generator_doc);
  
  Matrix references;
  const char *references_file_name = 
    IO::GetParam<std::string>("kernel/data").c_str();
  data::Load(references_file_name, &references);

  IO::Info << "nrows = " << references.n_rows() << " ncols = " << references.n_cols() << std::endl;

  const char* kernel_type = IO::GetParam<std::string.("kernel/kernel").c_str();
  int rate = IO::GetParam<int>("kernel/rate");

#define RUNNING 1
#if RUNNING==1  

  Matrix kernel_matrix;
  if (strcmp(kernel_type, "gaussian") == 0) 
    gen_gaussian_kernel(references, &kernel_matrix, rate);
  else if (strcmp(kernel_type, "polynomial") == 0) 
    gen_polynomial_kernel(references, &kernel_matrix, rate);

  /*
  // Kernel matrix to be outputted.
  Matrix kernel_matrix;

  // Initialize the kernel.
  GaussianKernel kernel;
  kernel.Init(bandwidth);

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
  */

  // Output the matrix.
  const char *file_name = IO::GetParam<std::string>("kernel/output").c_str();
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
#endif
  return 0;
}
