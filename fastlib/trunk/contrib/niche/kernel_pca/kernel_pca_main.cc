#include "kernel_pca.h"

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
  const char* data_filename = fx_param_str_req(NULL, "data");
  Matrix data;
  data::Load(data_filename, &data);
  
  KernelPCA kernel_pca;
  

  struct datanode* kpca_module =
    fx_submodule(fx_root, "kpca");


  kernel_pca.Init(data, kpca_module);

  Matrix K = kernel_pca.get_kernel_matrix();
  data::Save("kernel_matrix.csv", K);

  Vector eigenvalues;
  Matrix eigenvectors;
  kernel_pca.Compute(&eigenvalues, &eigenvectors);
  
  Matrix eigenvalues_matrix;
  eigenvalues_matrix.AliasColVector(eigenvalues);

  data::Save("kernel_eigenvalues.txt", eigenvalues_matrix);
  data::Save("kernel_eigenvectors.txt", eigenvectors);

  fx_done(fx_root);
}
