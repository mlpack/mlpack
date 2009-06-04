#include "fastlib/fastlib.h"
#include "kernel_pca.h"

int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  Matrix kernel_matrix;

  const char* filename = 
    fx_param_str(NULL, "kernelmat", "kde_kernel_matrix.csv");
  data::Load(filename, &kernel_matrix);

  Matrix kernel_principal_components;
  KernelPCA(kernel_matrix, &kernel_principal_components);

  data::Save("kpc.csv", kernel_principal_components);

  fx_done(fx_root);
}
