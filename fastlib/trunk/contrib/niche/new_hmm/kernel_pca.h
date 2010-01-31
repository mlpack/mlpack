#ifndef KERNEL_PCA_H
#define KERNEL_PCA_H


void KernelPCA(const Matrix &kernel_matrix,
	       Matrix *p_kernel_principal_components) {

  int n_points = kernel_matrix.n_rows();

  // center kernel matrix (center data in kernel space)
  Matrix averaging_matrix;
  averaging_matrix.Init(n_points, n_points);
  float inverse_m = ((double)1) / ((double)n_points);
  averaging_matrix.SetAll(inverse_m);
    
  Matrix centered_kernel_matrix;
  Matrix avg_by_kernel_matrix;
  Matrix avg_by_kernel_by_avg_matrix;
    
  la::MulInit(averaging_matrix, kernel_matrix, &avg_by_kernel_matrix);

  la::SubInit(avg_by_kernel_matrix, kernel_matrix, &centered_kernel_matrix);

  la::TransposeSquare(&avg_by_kernel_matrix);

  la::SubFrom(avg_by_kernel_matrix, &centered_kernel_matrix);
    
  la::MulInit(averaging_matrix, avg_by_kernel_matrix, &avg_by_kernel_by_avg_matrix);
    
  la::AddTo(avg_by_kernel_by_avg_matrix, &centered_kernel_matrix);
    
  
  // compute eigenvalues and eigenvectors of centered kernel matrix
  Vector eigenvalues;
  Matrix eigenvectors;

  Matrix right_singular_vectors;
  la::SVDInit(centered_kernel_matrix,
	      &eigenvalues, // singular values
	      &eigenvectors, // left singular vectors
	      &right_singular_vectors);

  for(int i = 0; i < n_points; i++) {
    Vector cur_eigenvector;
    eigenvectors.MakeColumnVector(i, &cur_eigenvector);
    la::Scale(1.0 / sqrt(eigenvalues[i]), &cur_eigenvector);
  }

  for(int i = 0; i < n_points; i++) {
    Vector cur_eigenvector;
    eigenvectors.MakeColumnVector(i, &cur_eigenvector);
    printf("%3e",
	   la::Dot(cur_eigenvector, cur_eigenvector) * eigenvalues[i]);
  }

  la::MulInit(centered_kernel_matrix, eigenvectors,
	      p_kernel_principal_components);
}


#endif /* KERNEL_PCA_H */
