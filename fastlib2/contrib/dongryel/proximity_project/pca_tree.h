#ifndef PCA_TREE_H
#define PCA_TREE_H

class PcaStat {
  
 private:

  static const double epsilon_ = 0.01;

  void AddVectorToMatrix(Matrix &A, const Vector &v, Matrix &R) {
    
    for(index_t i = 0; i < A.n_rows(); i++) {
      for(index_t j = 0; j < A.n_cols(); j++) {
	R.set(i, j, A.get(i, j) + v[i]);
      }
    }
  }

  void SubtractVectorFromMatrix(Matrix &A, const Vector &v, Matrix &R) {

    for(index_t i = 0; i < A.n_rows(); i++) {
      for(index_t j = 0; j < A.n_cols(); j++) {
	R.set(i, j, A.get(i, j) - v[i]);
      }
    }
  }

  void ComputeColumnSumVector(const Matrix &A, Vector &A_sum) {    
    A_sum.SetZero();
    for(index_t i = 0; i < A.n_cols(); i++) {
      Vector s;
      A.MakeColumnVector(i, &s);
      la::AddTo(s, &A_sum);
    }
  }

  void ComputeColumnMeanVector(const Matrix &A, int start, int count,
			       Vector &A_mean) {
    
    A_mean.SetZero();
    for(index_t i = 0; i < count; i++) {
      Vector s;
      A.MakeColumnVector(start + i, &s);
      la::AddTo(s, &A_mean);
    }
    la::Scale(1.0 / ((double) count), &A_mean);
  }

  void ComputeColumnMeanVector(const Matrix &A, Vector &A_mean) {

    A_mean.SetZero();
    for(index_t i = 0; i < A.n_cols(); i++) {
      Vector s;
      A.MakeColumnVector(i, &s);
      la::AddTo(s, &A_mean);
    }
    la::Scale(1.0 / ((double) A.n_cols()), &A_mean);
  }

  void ExtractSubMatrix(const Matrix &source, int start, int count,
			Matrix &dest) {

    // copy from source to target, while computing the mean coordinates
    for(int i = start; i < start + count; i++) {
      Vector s, t;
      source.MakeColumnVector(i, &s);
      dest.MakeColumnVector(i - start, &t);
      t.CopyValues(s);
    }
  }

  void ExtractSubMatrix(Matrix &source, int start, 
			int count, Matrix &dest) {
    
    // copy from source to target, while computing the mean coordinates
    for(int i = start; i < start + count; i++) {
      Vector s, t;
      source.MakeColumnVector(i, &s);
      dest.MakeColumnVector(i - start, &t);
      t.CopyValues(s);
    }
  }

 public:

  int start_;
  
  int count_;
  
  Vector means_;

  Matrix eigenvectors_;

  Matrix eigenvalues_;

  /** Initialize the statistics */
  void Init() {
  }

  /** compute PCA exhaustively for leaf nodes */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {

    Matrix mean_centered_;
    means_.Destruct();
    eigenvectors_.Destruct();
    eigenvalues_.Destruct();

    // set the starting index and the count
    start_ = start;
    count_ = count;

    // copy the mean-centered submatrix of the points in this leaf node
    mean_centered_.Init(dataset.n_rows(), count);
    means_.Init(dataset.n_rows());

    // extract the relevant part of the dataset and mean-center it
    ExtractSubMatrix(dataset, start, count, mean_centered_);

    ComputeColumnMeanVector(mean_centered_, means_);
    SubtractVectorFromMatrix(mean_centered_, means_, mean_centered_);

    // compute PCA on the extracted submatrix
    Matrix right_singular_vectors_transposed, left_singular_vectors;
    Vector singular_values;

    la::SVDInit(mean_centered_, &singular_values, 
		&left_singular_vectors, &right_singular_vectors_transposed);

    // find out how many eigenvalues to keep
    int eigencount = 0;
    double max_singular_value = 0;
    for(index_t i = 0; i < singular_values.length(); i++) {
      if(singular_values[i] > max_singular_value) {
	max_singular_value = singular_values[i];
      }
    }
    for(index_t i = 0; i < singular_values.length(); i++) {
      if(singular_values[i] >= epsilon_ * max_singular_value) {
	eigencount++;
      }
    }

    eigenvalues_.Init(eigencount, eigencount);
    eigenvalues_.SetZero();
    eigenvectors_.Init(dataset.n_rows(), eigencount);

    // relationship between the singular value and the eigenvalue is
    // enforced here
    for(index_t i = 0, index = 0; i < singular_values.length(); i++) {
      if(singular_values[i] > epsilon_ * max_singular_value) {
	Vector source, destination;
	eigenvalues_.set(index, index, 
			 singular_values[i] * singular_values[i] / 
			 ((double) count_));

	left_singular_vectors.MakeColumnVector(i, &source);
	eigenvectors_.MakeColumnVector(index, &destination);
	destination.CopyValues(source);
	index++;
      }
    }
  }

  /**
   * Takes the two eigenbases and 
   */
  void ComputeLeftsideNullSpaceBasis(const PcaStat& left_stat,
				     const PcaStat& right_stat, 
				     Matrix *leftside_nullspace_basis,
				     Matrix *projection_of_right_eigenbasis,
				     Vector *mean_diff,
				     Vector *projection_of_mean_diff) {

    Matrix left_eigenbasis, right_eigenbasis;
    Vector left_means, right_means;
    Matrix projection_residue_of_right_eigenbasis;
    Vector projection_residue_of_mean_diff;

    // left and right's eigenbasis and the mean vectors
    left_eigenbasis.Alias(left_stat.eigenvectors_);
    right_eigenbasis.Alias(right_stat.eigenvectors_);
    left_means.Alias(left_stat.means_);
    right_means.Alias(right_stat.means_);

    // compute mean difference between two eigenspaces
    la::SubInit(right_means, left_means, mean_diff);

    // compute the projection of the right eigenbasis onto the left
    // eigenbasis
    la::MulTransAInit(left_eigenbasis, right_eigenbasis, 
		      projection_of_right_eigenbasis);

    // compute the residue of projection
    projection_residue_of_right_eigenbasis.Copy(right_eigenbasis);
    la::MulExpert(-1, left_eigenbasis, *projection_of_right_eigenbasis, 1,
		  &projection_residue_of_right_eigenbasis);
    
    // project the difference in the two means onto the eigenbasis
    // belonging to the first subspace
    la::MulInit(*mean_diff, left_eigenbasis, projection_of_mean_diff);
    
    // compute the projection residue of the mean difference
    projection_residue_of_mean_diff.Copy(*mean_diff);
    la::MulExpert(-1, left_eigenbasis, *projection_of_mean_diff, 1,
		  &projection_residue_of_mean_diff);
    
    // copy onto a temporary matrix for QR decomposition by filtering
    // very small columns
    Matrix span_set;
    int dim = left_eigenbasis.n_rows();
    span_set.Init(dim, 0);
    
    // loop over each column residue vectors
    for(index_t i = 0; i < projection_residue_of_right_eigenbasis.n_cols(); 
	i++) {
      Vector residue_vector;
      double euclidean_norm;
      projection_residue_of_right_eigenbasis.MakeColumnVector
	(i, &residue_vector);
      euclidean_norm = la::LengthEuclidean(residue_vector);

      if(euclidean_norm > epsilon_) {
	Vector dest;
	span_set.ResizeNoalias(span_set.n_cols() + 1);
	span_set.MakeColumnVector(span_set.n_cols() - 1, &dest);
	dest.CopyValues(residue_vector);
      }
    }
    double euclidean_norm_of_projection_residue_of_mean_diff =
      la::LengthEuclidean(projection_residue_of_mean_diff);
    if(euclidean_norm_of_projection_residue_of_mean_diff > epsilon_) {
      Vector dest;
      span_set.ResizeNoalias(span_set.n_cols() + 1);
      span_set.MakeColumnVector(span_set.n_cols() - 1, &dest);
      dest.CopyValues(projection_residue_of_mean_diff);
    }

    if(span_set.n_cols() > 0) {
      Matrix dummy;
      la::QRInit(span_set, leftside_nullspace_basis, &dummy);
    }
    else {
      leftside_nullspace_basis->Init(dim, 0);
    }
  }

  void SetupEigensystem(const PcaStat& left_stat, const PcaStat &right_stat, 
			const Matrix &leftside_nullspace_basis,
			const Matrix &projection_of_right_eigenbasis,
			const Vector &mean_diff, 
			const Vector &projection_of_mean_diff, 
			Matrix *eigensystem) {

    Matrix left_eigenbasis, right_eigenbasis, 
      left_eigenvalues, right_eigenvalues;

    // left and right's eigenbasis and the mean vectors
    left_eigenbasis.Alias(left_stat.eigenvectors_);
    right_eigenbasis.Alias(right_stat.eigenvectors_);
    left_eigenvalues.Alias(left_stat.eigenvalues_);
    right_eigenvalues.Alias(right_stat.eigenvalues_);

    // precomputed factors
    double factor1 = ((double) left_stat.count_) / ((double) count_);
    double factor2 = ((double) right_stat.count_) / ((double) count_);
    double factor3 = ((double) left_stat.count_ * right_stat.count_) /
      ((double) count_ * count_);

    if(leftside_nullspace_basis.n_cols() > 0) {
      eigensystem->Init(left_eigenvalues.n_rows() + 
			leftside_nullspace_basis.n_cols(),
			left_eigenvalues.n_rows() +
			leftside_nullspace_basis.n_cols());
    }
    else {
      eigensystem->Init(left_eigenvalues.n_rows(), left_eigenvalues.n_rows());
    }
    eigensystem->SetZero();
    
    // compute the top left part of the eigensystem
    Matrix top_left, top_tmp;
    la::MulInit(projection_of_right_eigenbasis, right_eigenvalues, &top_tmp);
    la::MulTransBInit(top_tmp, projection_of_right_eigenbasis, &top_left);

    for(index_t i = 0; i < left_eigenvalues.n_rows(); i++) {
      for(index_t j = 0; j < left_eigenvalues.n_cols(); j++) {
	eigensystem->set(i, j, factor1 * left_eigenvalues.get(i, j) +
			 factor2 * top_left.get(i, j) +
			 factor3 * projection_of_mean_diff[i] *
			 projection_of_mean_diff[j]);
      }
    }

    // handling the top right, bottom left, and bottom right
    if(leftside_nullspace_basis.n_cols() > 0) {
      Matrix proj_rightside_eigenbasis_on_leftside_nullspace;
      Vector proj_mean_diff_on_leftside_nullspace;
      la::MulTransAInit(leftside_nullspace_basis, right_eigenbasis,
			&proj_rightside_eigenbasis_on_leftside_nullspace);
      la::MulInit(mean_diff, leftside_nullspace_basis,
		  &proj_mean_diff_on_leftside_nullspace);
      
      // set up the eigensystem
      Matrix top_right, bottom_left, bottom_right;
      Matrix bottom_tmp;
     
      la::MulTransBInit(top_tmp, 
			proj_rightside_eigenbasis_on_leftside_nullspace,
			&top_right);
      for(index_t i = 0; i < top_right.n_rows(); i++) {
	for(index_t j = 0; j < top_right.n_cols(); j++) {
	  eigensystem->set(i, j + top_left.n_cols(),
			   factor2 * top_right.get(i, j) +
			   factor3 * projection_of_mean_diff[i] *
			   proj_mean_diff_on_leftside_nullspace[j]);
	}
      }

      la::MulInit(proj_rightside_eigenbasis_on_leftside_nullspace,
		  right_eigenvalues, &bottom_tmp);
      la::MulTransBInit(bottom_tmp, projection_of_right_eigenbasis, 
			&bottom_left);
      for(index_t i = 0; i < bottom_left.n_rows(); i++) {
	for(index_t j = 0; j < bottom_left.n_cols(); j++) {
	  eigensystem->set(i + top_left.n_rows(), j,
			   factor2 * bottom_left.get(i, j) +
			   factor3 * proj_mean_diff_on_leftside_nullspace[i] *
			   projection_of_mean_diff[j]);
	}
      }
      la::MulTransBInit(bottom_tmp, 
			proj_rightside_eigenbasis_on_leftside_nullspace,
			&bottom_right);
      for(index_t i = 0; i < bottom_right.n_rows(); i++) {
	for(index_t j = 0; j < bottom_right.n_cols(); j++) {
	  eigensystem->set(i + top_left.n_rows(), j + top_left.n_cols(),
			   factor2 * bottom_right.get(i, j) +
			   factor3 * proj_mean_diff_on_leftside_nullspace[i] *
			   proj_mean_diff_on_leftside_nullspace[j]);
	}
      }
    } // end of case for nonempty leftside nullspace
  }

  /** 
   * Merge two eigenspaces into one
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const PcaStat& left_stat, const PcaStat& right_stat) {

    means_.Destruct();
    eigenvectors_.Destruct();
    eigenvalues_.Destruct();

    // set up starting index and the count
    start_ = start;
    count_ = count;

    Matrix leftside_nullspace_basis, projection_of_right_eigenbasis;
    Vector mean_diff, projection_of_mean_diff;
    Matrix eigensystem;

    ComputeLeftsideNullSpaceBasis(left_stat, right_stat, 
				  &leftside_nullspace_basis,
				  &projection_of_right_eigenbasis,
				  &mean_diff, &projection_of_mean_diff);

    SetupEigensystem(left_stat, right_stat,
		     leftside_nullspace_basis,
		     projection_of_right_eigenbasis,
		     mean_diff, projection_of_mean_diff, &eigensystem);

    // compute eigenvalues and eigenvectors of the system
    Matrix rotation, combined_subspace;
    Vector evalues;
    la::EigenvectorsInit(eigensystem, &evalues, &rotation);
    combined_subspace.Copy(left_stat.eigenvectors_);
    combined_subspace.ResizeNoalias(combined_subspace.n_cols() + 
				    leftside_nullspace_basis.n_cols());
    for(index_t i = 0; i < leftside_nullspace_basis.n_cols(); i++) {
      Vector source, dest;
      leftside_nullspace_basis.MakeColumnVector(i, &source);
      combined_subspace.MakeColumnVector(i + left_stat.eigenvectors_.n_cols(), 
					 &dest);
      dest.CopyValues(source);
    }

    // rotate the left eigenbasis plus the null space of the leftside to
    // get the global eigenbasis
    la::MulInit(combined_subspace, rotation, &eigenvectors_);

    // find out how many eigenvalues to keep
    int eigencount = 0;
    double max_eigenvalue = 0;
    for(index_t i = 0; i < evalues.length(); i++) {
      if(evalues[i] > max_eigenvalue) {
	max_eigenvalue = evalues[i];
      }
    }
    for(index_t i = 0; i < evalues.length(); i++) {
      if(evalues[i] > epsilon_ * max_eigenvalue) {
	eigencount++;
      }
    }
    eigenvalues_.Init(eigencount, eigencount);
    eigenvalues_.SetZero();

    // relationship between the singular value and the eigenvalue is
    // enforced here
    for(index_t i = 0, index = 0; i < evalues.length(); i++) {
      if(evalues[i] > epsilon_ * max_eigenvalue) {
	Vector s, d;
	eigenvalues_.set(index, index, evalues[i]);
	eigenvectors_.MakeColumnVector(i, &s);
	eigenvectors_.MakeColumnVector(index, &d);
	d.CopyValues(s);
	index++;
      }
    }
    eigenvectors_.ResizeNoalias(eigencount);

    // compute the weighted average of the two means
    double factor1 = ((double) left_stat.count_) / ((double) count_);
    double factor2 = ((double) right_stat.count_) / ((double) count_);
    means_.Copy(left_stat.means_);
    la::Scale(factor1, &means_);
    la::AddExpert(factor2, right_stat.means_, &means_);
  }

  PcaStat() {
    means_.Init(1);
    eigenvectors_.Init(1, 1);
    eigenvalues_.Init(1, 1);
  }

  ~PcaStat() { }

};

#endif
