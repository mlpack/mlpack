#ifndef SUBSPACE_STAT_H
#define SUBSPACE_STAT_H

class SubspaceStat {
  
 private:

  static const double epsilon_ = 0.01;

 public:

  int start_;
  
  int count_;
  
  Vector means_;

  Matrix eigenvectors_;

  /** compute PCA exhaustively for leaf nodes */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {

    // Degenerate case: the leaf node contains only one point...
    if(count == 1) {
      Vector point;
      dataset.MakeColumnVector(start, &point);
      means_.Copy(point);
      eigenvectors_.Init(dataset.n_rows(), 1);
      eigenvectors_.SetZero();
      return;
    }

    Matrix mean_centered;

    // set the starting index and the count
    start_ = start;
    count_ = count;

    // copy the mean-centered submatrix of the points in this leaf node
    means_.Init(dataset.n_rows());

    // extract the relevant part of the dataset and mean-center it
    Pca::ComputeMean(dataset, start, count, &means_);
    Pca::MeanCenter(dataset, start, count, means_, &mean_centered);

    // compute PCA on the extracted submatrix
    Matrix right_singular_vectors_transposed, left_singular_vectors;
    Vector singular_values;

    la::SVDInit(mean_centered, &singular_values, 
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
    eigenvectors_.Init(dataset.n_rows(), eigencount);

    // relationship between the singular value and the eigenvalue is
    // enforced here
    for(index_t i = 0, index = 0; i < singular_values.length(); i++) {
      if(singular_values[i] >= epsilon_ * max_singular_value) {
	Vector source, destination;

	left_singular_vectors.MakeColumnVector(i, &source);
	eigenvectors_.MakeColumnVector(index, &destination);
	destination.CopyValues(source);
	index++;
      }
    }
  }

  /** 
   * Merge two eigenspaces into one
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const SubspaceStat& left_stat, const SubspaceStat& right_stat) {

    // set up starting index and the count
    start_ = start;
    count_ = count;

    // compute the weighted average of the two means
    double factor1 = ((double) left_stat.count_) / ((double) count_);
    double factor2 = ((double) right_stat.count_) / ((double) count_);
    means_.Copy(left_stat.means_);
    la::Scale(factor1, &means_);
    la::AddExpert(factor2, right_stat.means_, &means_);
  }

  SubspaceStat() { }

  ~SubspaceStat() { }
  
};

#endif
