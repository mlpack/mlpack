#ifndef PCA_TREE_H
#define PCA_TREE_H

class PCAStat {
  
 private:

  inline success_t FullSVDInit(const Matrix &A, Vector *s, Matrix *U, 
			       Matrix *VT) {
    f77_integer k = min(A.n_rows(), A.n_cols());
    s->Init(k);
    U->Init(A.n_rows(), A.n_rows());
    VT->Init(A.n_cols(), A.n_cols());
    Matrix tmp;
    tmp.Copy(A);
    
    f77_integer info;
    f77_integer m = tmp.n_rows();
    f77_integer n = tmp.n_cols();
    f77_integer iwork[8 * k];
    const char *job = "A";
    double d; // for querying optimal work size
    
    F77_FUNC(dgesdd)(job, m, n, tmp.ptr(), m,
		     s->ptr(), U->ptr(), m, VT->ptr(), n, &d, -1, 
		     iwork, &info);
    {
      f77_integer lwork = (f77_integer)d;
      // work for DGESDD can be large, we really do need to malloc it
      double *work = mem::Alloc<double>(lwork);
      
      F77_FUNC(dgesdd)(job, m, n, tmp.ptr(), m,
		       s->ptr(), U->ptr(), m, VT->ptr(), n, work, 
		       lwork, iwork, &info);
      
      mem::Free(work);
    }

    return SUCCESS_FROM_LAPACK(info);
  }

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
  
  Matrix mean_centered_;

  Matrix pca_transformed_;
  
  Vector means_;

  /** Initialize the statistics */
  void Init() {
  }

  /** compute PCA exhaustively for leaf nodes */
  void Init(const Matrix& dataset, index_t &start, index_t &count) {

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
    Matrix U, VT;
    Vector s_values;
    la::SVDInit(mean_centered_, &s_values, &U, &VT);

    // reduce the dimension in half
    Matrix U_trunc;
    la::TransposeInit(U, &U_trunc);

    // transform coordinates
    la::MulInit(U_trunc, mean_centered_, &pca_transformed_);

    printf("Leaf exhaustive:\n");
    pca_transformed_.PrintDebug();
  }

  /** 
   * domain-decomposition based PCA merging using two-sided affine
   * transformation
   */
  void Init(const Matrix& dataset, index_t &start, index_t &count,
	    const PCAStat& left_stat, const PCAStat& right_stat) {

    int dim = dataset.n_rows();

    double factor1 = ((double) left_stat.count_ + right_stat.count_) /
      ((double) count);
    double factor2 = 1.0 - factor1;
    double factor3 = sqrt(((double) left_stat.count_ * right_stat.count_) /
			  ((double) left_stat.count_ + right_stat.count_));

    // set the starting index and the number of points in this node
    start_ = start;
    count_ = count;

    // compute the combined mean using the left and the right means    
    means_.Copy(left_stat.means_);
    la::Scale(factor1, &means_);
    la::AddExpert(factor2, right_stat.means_, &means_);
    
    // span that includes the difference in the two mean vectors and
    // the subspace spanned by the right node
    Matrix perturbation;
    perturbation.Init(dim, right_stat.count_ + 1);
    for(int i = 0; i < dim; i++) {

      for(int j = 0; j < right_stat.count_; j++) {
	perturbation.set(i, j, right_stat.mean_centered_.get(i, j) +
			 right_stat.means_[i] - means_[i]);
      }
      perturbation.set(right_stat.count_, i, factor3 * 
		       (left_stat.means_[i] - right_stat.means_[i]));
    }
    
    // apply QR decomposition to get orthonormal basis of the span above
    Matrix q, r;
    la::QRInit(perturbation, &q, &r);

    // combined system
    
  }

  PCAStat() { }

  ~PCAStat() { }

};

#endif
