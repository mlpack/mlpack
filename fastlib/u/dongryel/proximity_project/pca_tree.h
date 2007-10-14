#ifndef PCA_TREE_H
#define PCA_TREE_H

class PCAStat {
  
 private:

  void SubtractVectorFromMatrix(Matrix &A, const Vector &v, Matrix &R) {

    for(index_t i = 0; i < A.n_rows(); i++) {
      for(index_t j = 0; j < A.n_cols(); j++) {
	R.set(i, j, A.get(i, j) - v[i]);
      }
    }
  }

  void PseudoInverse(const Matrix &A, Matrix *A_inv) {
    Vector ro_s;
    Matrix ro_U, ro_VT;

    // compute the SVD of A
    la::SVDInit(A, &ro_s, &ro_U, &ro_VT);
    
    // take the transpose of V^T and U
    Matrix ro_VT_trans;
    Matrix ro_U_trans;
    la::TransposeInit(ro_VT, &ro_VT_trans);
    la::TransposeInit(ro_U, &ro_U_trans);
    Matrix ro_s_inv;
    ro_s_inv.Init(ro_VT_trans.n_cols(), ro_U_trans.n_rows());
    ro_s_inv.SetZero();

    // initialize the diagonal by the inverse of ro_s
    for(index_t i = 0; i < ro_s.length(); i++) {
      ro_s_inv.set(i, i, 1.0 / ro_s[i]);
    }
    Matrix intermediate;
    la::MulInit(ro_s_inv, ro_U_trans, &intermediate);
    la::MulInit(ro_VT_trans, intermediate, A_inv);
    
    printf("Testing pseudoinverse\n");
    ro_s.PrintDebug();
    ro_s_inv.PrintDebug();
    A.PrintDebug();
    A_inv->PrintDebug();

    Matrix check;
    la::MulInit(A, *A_inv, &check);
    check.PrintDebug();
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

  int begin_;
  
  int count_;
  
  Matrix orig_mean_centered_;

  Matrix pca_transformed_;
  
  Vector means_;

  /** Initialize the statistics */
  void Init() {
    orig_mean_centered_.Init(0, 0);
    pca_transformed_.Init(0, 0);
    means_.Init(0);
  }

  /** compute PCA exhaustively for leaf nodes */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {

    printf("Base case...\n");
    printf("Got %d %d\n", start, count);

    begin_ = start;
    count_ = count;

    // copy the mean-centered submatrix of the points in this leaf node
    orig_mean_centered_.Destruct();
    pca_transformed_.Destruct();
    means_.Destruct();

    orig_mean_centered_.Init(dataset.n_rows(), count);
    means_.Init(dataset.n_rows());

    // extract the relevant part of the dataset and mean-center it
    ExtractSubMatrix(dataset, start, count, orig_mean_centered_);
    ComputeColumnMeanVector(orig_mean_centered_, means_);
    SubtractVectorFromMatrix(orig_mean_centered_, means_, orig_mean_centered_);

    // compute PCA on the extracted submatrix
    Matrix U, VT;
    Vector s;
    la::SVDInit(orig_mean_centered_, &s, &U, &VT);

    // reduce the dimension in half
    Matrix U_trunc;
    int new_dimension = U.n_cols();
    U_trunc.Init(new_dimension, U.n_rows());
    for(index_t i = 0; i < new_dimension; i++) {
      Vector s;
      U.MakeColumnVector(i, &s);
      
      for(index_t j = 0; j < U.n_rows(); j++) {
	U_trunc.set(i, j, s[j]);
      }
    }

    la::MulInit(U_trunc, orig_mean_centered_, &pca_transformed_);

    means_.PrintDebug();
    pca_transformed_.PrintDebug();

    /*
    // this shows how to recover global coordinates
    Matrix recovered_;
    la::MulTransAInit(U_trunc, pca_transformed_, &recovered_);

    printf("Original mean centered...\n");
    orig_mean_centered_.PrintDebug();
    printf("Recovered...\n");
    recovered_.PrintDebug();
    */
  }

  /** 
   * domain-decomposition based PCA merging using one-sided affine
   * transformation
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const PCAStat& left_stat, const PCAStat& right_stat) {

    printf("Merging case...\n\n");

    begin_ = start;
    count_ = count;

    printf("Got %d %d\n", start, count);

    // destory objects and reinitialize
    orig_mean_centered_.Destruct();
    pca_transformed_.Destruct();
    means_.Destruct();
    orig_mean_centered_.Init(dataset.n_rows(), count);
    means_.Init(dataset.n_rows());

    // compute the entire mean and the difference between the left and
    // the right means
    ComputeColumnMeanVector(dataset, start, count, means_);

    // copy the two overlap regions
    Matrix left_pca_transformed;
    Matrix right_pca_transformed;
    Matrix left_overlap;
    Matrix right_overlap;
    Matrix left_overlap_meancentered;
    Matrix right_overlap_meancentered;
    left_pca_transformed.Alias(left_stat.pca_transformed_);
    right_pca_transformed.Alias(right_stat.pca_transformed_);

    int num_overlap = left_stat.begin_ + left_stat.count_ -
      right_stat.begin_;
    left_overlap.Init(left_pca_transformed.n_rows(),num_overlap);
    right_overlap.Init(right_pca_transformed.n_rows(), num_overlap);
    left_overlap_meancentered.Init(left_pca_transformed.n_rows(),
				   num_overlap);
    right_overlap_meancentered.Init(right_pca_transformed.n_rows(),
				    num_overlap);
    ExtractSubMatrix(left_pca_transformed, 
		     left_pca_transformed.n_cols() - num_overlap,
		     num_overlap, left_overlap);
    ExtractSubMatrix(right_pca_transformed, 0, num_overlap, right_overlap);

    // compute the column means of the left overlap and the right overlap
    Vector left_overlap_means;
    Vector right_overlap_means;
    left_overlap_means.Init(left_overlap.n_rows());
    right_overlap_means.Init(right_overlap.n_rows());
    ComputeColumnMeanVector(left_overlap, left_overlap_means);
    ComputeColumnMeanVector(right_overlap, right_overlap_means);

    // compute mean centered left and right overlap regions
    SubtractVectorFromMatrix(left_overlap, left_overlap_means,
			     left_overlap_meancentered);
    SubtractVectorFromMatrix(right_overlap, right_overlap_means,
			     right_overlap_meancentered);

    // get the pseudoinverse of the mean-centered right overlap
    Matrix right_overlap_meancentered_inv;
    PseudoInverse(right_overlap_meancentered, &right_overlap_meancentered_inv);

    // this is the transformation matrix from right overlap to left overlap
    Matrix F;
    la::MulInit(left_overlap_meancentered, right_overlap_meancentered_inv,
		&F);

    printf("F is\n");
    F.PrintDebug();


    // use the transformation matrix to map the right side local coordinate
    // to the left side local coordinate. left local coordinates that
    // are not in the overlap region stay the same
    pca_transformed_.Init(left_pca_transformed.n_rows(), count);
    pca_transformed_.SetZero();
    for(index_t i = 0; i < left_pca_transformed.n_cols() - num_overlap; i++) {
      Vector sl, d;
      left_pca_transformed.MakeColumnVector(i, &sl);
      pca_transformed_.MakeColumnVector(i, &d);
      d.CopyValues(sl);
    }
    Vector right_overlap_means_transformed;
    la::MulInit(F, right_overlap_means, &right_overlap_means_transformed);
    for(index_t i = 0; i < right_pca_transformed.n_rows(); i++) {
      for(index_t j = 0; j < right_pca_transformed.n_cols(); j++) {
	double dotprod = 0;
	for(index_t k = 0; k < F.n_cols(); k++) {
	  dotprod += F.get(i, k) * right_pca_transformed.get(k, j);
	}
	dotprod += (left_overlap_means[i] - 
		    right_overlap_means_transformed[i]);
	pca_transformed_.set(i, pca_transformed_.n_cols() - 		     
			     right_pca_transformed.n_cols() + j, dotprod);
      }
    }
    
    printf("Final output!\n");
    printf("Left: %d %d\n", left_stat.begin_, left_stat.count_);
    printf("Right: %d %d\n", right_stat.begin_, right_stat.count_);
    left_pca_transformed.PrintDebug();
    right_pca_transformed.PrintDebug();
    pca_transformed_.PrintDebug();

    printf("Checking!\n");
    Init(dataset, start, count);
    exit(0);
  }

  PCAStat() { Init(); }

  ~PCAStat() { }

};

#endif
