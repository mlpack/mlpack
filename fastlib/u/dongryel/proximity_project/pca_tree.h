#ifndef PCA_TREE_H
#define PCA_TREE_H

class PCAStat {
  
 private:
  
  void ExtractSubMatrix(const Matrix &source, index_t &start, 
			index_t &count) {
    
    // copy from source to target, while computing the mean coordinates
    means_.SetZero();
    for(index_t i = start; i < start + count; i++) {
      Vector s, t;
      source.MakeColumnVector(i, &s);
      orig_mean_centered_.MakeColumnVector(i - start, &t);
      t.CopyValues(s);
      la::AddTo(s, &means_);
    }
    la::Scale(1.0 / ((double) count), &means_);

    // go through the target and mean-center it
    for(index_t i = 0; i < count; i++) {
      Vector s;
      orig_mean_centered_.MakeColumnVector(i, &s);
      la::SubFrom(means_, &s);
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

    begin_ = start;
    count_ = count;

    // copy the mean-centered submatrix of the points in this leaf node
    orig_mean_centered_.Destruct();
    pca_transformed_.Destruct();
    means_.Destruct();

    orig_mean_centered_.Init(dataset.n_rows(), count);
    means_.Init(dataset.n_rows());

    ExtractSubMatrix(dataset, start, count);
    
    // compute PCA on the extracted submatrix
    Matrix U, VT;
    Vector s;
    la::SVDInit(orig_mean_centered_, &s, &U, &VT);

    // reduce the dimension in half
    Matrix U_trunc;
    int new_dimension = U.n_cols() / 2;
    U_trunc.Init(new_dimension, U.n_rows());
    for(index_t i = 0; i < new_dimension; i++) {
      Vector s;
      U.MakeColumnVector(i, &s);
      
      for(index_t j = 0; j < U.n_rows(); j++) {
	U_trunc.set(i, j, s[j]);
      }
    }
    
    U_trunc.PrintDebug();

    la::MulInit(U_trunc, orig_mean_centered_, &pca_transformed_);
    pca_transformed_.PrintDebug();

    /*
      // this shows how to recover global coordinates
    Matrix recovered_;
    la::MulTransAInit(U_trunc, pca_transformed_, &recovered_);

    orig_mean_centered_.PrintDebug();
    recovered_.PrintDebug();
    */
  }

  /** domain-decomposition based PCA merging */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const PCAStat& left_stat, const PCAStat& right_stat) {
    begin_ = start;
    count_ = count;

    orig_mean_centered_.Destruct();
    pca_transformed_.Destruct();
    means_.Destruct();

    orig_mean_centered_.Init(dataset.n_rows(), count);
    means_.Init(dataset.n_rows());

    // determine the overlapped points between the children
    printf("Parent: %d %d\n", start, count);
    printf("Left: %d %d\n", left_stat.begin_, 
	   left_stat.begin_ + left_stat.count_);
    printf("Right: %d %d\n", right_stat.begin_, 
	   right_stat.begin_ + right_stat.count_);

    // copy the two overlap regions
    Matrix left_pca_transformed;
    Matrix right_pca_transformed;
    Matrix left_overlap;
    Matrix right_overlap;
    left_pca_transformed.Alias(left_stat.pca_transformed_);
    right_pca_transformed.Alias(right_stat.pca_transformed_);

    int num_overlap = left_stat.begin_ + left_stat.count_ -
      right_stat.begin_;

    left_overlap.Init(left_pca_transformed.n_rows(),num_overlap);
    right_overlap.Init(right_pca_transformed.n_rows(), num_overlap);
    
    for(index_t i = 0; i < num_overlap; i++) {
      Vector sl, sr, dl, dr;

      left_pca_transformed.MakeColumnVector(left_pca_transformed.n_cols() -
					    num_overlap + i, &sl);
      right_pca_transformed.MakeColumnVector(i, &sr);
      left_overlap.MakeColumnVector(i, &dl);
      right_overlap.MakeColumnVector(i, &dr);
      
      dl.CopyValues(sl);
      dr.CopyValues(sr);
    }
    
    left_pca_transformed.PrintDebug();
    right_pca_transformed.PrintDebug();
    left_overlap.PrintDebug();
    right_overlap.PrintDebug();
    exit(0);
  }

  PCAStat() { Init(); }

  ~PCAStat() { }

};

#endif
