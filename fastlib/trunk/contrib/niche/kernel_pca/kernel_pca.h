#include <fastlib/fastlib.h>

class KernelPCA {

 private:

  struct datanode* module_;

  // the data
  //Matrix data_;
  
  // the kernel matrix
  Matrix kernel_matrix_;

  // the number of points
  int num_points_;

  // the kernel function's bandwidth
  double bandwidth_;



 public:

  void Init(const Matrix& data_in, struct datanode* module_in) {

    module_ = module_in;

    num_points_ = data_in.n_cols();

    // compute kernel matrix

    kernel_matrix_.Init(num_points_, num_points_);

    bandwidth_ = fx_param_double_req(module_, "h");
    DEBUG_ASSERT(bandwidth_ > 0);

    GaussianKernel gaussian_kernel;
    gaussian_kernel.Init(bandwidth_);

    for(int j = 0; j < num_points_; j++) {
      Vector x_j;
      data_in.MakeColumnVector(j, &x_j);

      for(int i = 0; i < num_points_; i++) {
	Vector x_i;
	data_in.MakeColumnVector(i, &x_i);
	
	double sq_dist = la::DistanceSqEuclidean(x_i, x_j);

	kernel_matrix_.set(i, j, gaussian_kernel.EvalUnnormOnSq(sq_dist));
      }
    }

    la::Scale(gaussian_kernel.CalcNormConstant(data_in.n_rows()), &kernel_matrix_);


  }

  Matrix get_kernel_matrix() {
    return kernel_matrix_;
  }



  void Compute(Vector* p_eigenvalues, Matrix* p_eigenvectors) {
    

    // center kernel matrix (center data in kernel space)

    Matrix averaging_matrix;
    averaging_matrix.Init(num_points_, num_points_);
    float inverse_m = ((double)1) / ((double)num_points_);
    averaging_matrix.SetAll(inverse_m);
    
    Matrix centered_kernel_matrix;
    Matrix avg_by_kernel_matrix;
    Matrix avg_by_kernel_by_avg_matrix;
    
    la::MulInit(averaging_matrix, kernel_matrix_, &avg_by_kernel_matrix);

    la::SubInit(avg_by_kernel_matrix, kernel_matrix_, &centered_kernel_matrix);

    la::TransposeSquare(&avg_by_kernel_matrix);

    la::SubFrom(avg_by_kernel_matrix, &centered_kernel_matrix);
    
    la::MulInit(averaging_matrix, avg_by_kernel_matrix, &avg_by_kernel_by_avg_matrix);
    
    la::AddTo(avg_by_kernel_by_avg_matrix, &centered_kernel_matrix);
    
    
    // compute eigenvalues and eigenvectors of centered kernel matrix
    Matrix right_singular_vectors;
    la::SVDInit(centered_kernel_matrix,
		p_eigenvalues, // singular values
		p_eigenvectors, // left singular vectors
		&right_singular_vectors);
    //la::EigenvectorsInit(centered_kernel_matrix, eigenvalues, eigenvectors);
    
    const Matrix &eigenvectors = *p_eigenvectors;
    const Vector &eigenvalues = *p_eigenvalues;
    
    
    for(int i = 0; i < num_points_; i++) {
      Vector cur_eigenvector;
      eigenvectors.MakeColumnVector(i, &cur_eigenvector);
      la::Scale(1.0 / sqrt(eigenvalues[i]), &cur_eigenvector);
    }
    
    for(int i = 0; i < num_points_; i++) {
      Vector cur_eigenvector;
      eigenvectors.MakeColumnVector(i, &cur_eigenvector);
      printf("%3e\n",
	     la::Dot(cur_eigenvector, cur_eigenvector) * eigenvalues[i]);
    }
  }
  
};
