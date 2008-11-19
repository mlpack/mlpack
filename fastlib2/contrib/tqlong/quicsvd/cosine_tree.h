#ifndef QUICSVD_COSINE_TREE_H
#define QUICSVD_COSINE_TREE_H
#include <fastlib/fastlib.h>

class CosineNode {
  /** The matrix to be approximated A ~ A' = U S VT*/
  Matrix A_;
  ArrayList<int> origIndices_;
  ArrayList<double> norms_;
  ArrayList<double> cum_norms_;
  Vector mean_;
  bool isLeft_;
  double L2Err_;

  CosineNode *parent_, *left_, *right_;

  OT_DEF(CosineNode) {
    OT_MY_OBJECT(A_);
    OT_MY_OBJECT(origIndices_);
    OT_MY_OBJECT(norms_);
    OT_MY_OBJECT(cum_norms_);
    OT_MY_OBJECT(mean_);
    OT_MY_OBJECT(isLeft_);
  }

 public:
  CosineNode(const Matrix& A);
  CosineNode(CosineNode& parent, const ArrayList<int>& indices,
	     bool isLeft);

  void GetColumn(int i_col, Vector* col) {
    A_.MakeColumnVector(origIndices_[i_col], col);
  }

  index_t n_cols() const {
    return origIndices_.size();
  }

  void Split();
  
  double getSumL2() const {
    return cum_norms_[n_cols()-1];
  }

  const Vector& getMean() const {
    return mean_;
  }

  index_t getOrigIndex(index_t i_col) const {
    return origIndices_[i_col];
  }

  void setL2Err(double L2Err) {
    L2Err_ = L2Err;
  }

  bool hasLeft() const {
    return left_ != NULL;
  }

  bool hasRight() const {
    return right_ != NULL;
  }

  CosineNode* getLeft() {
    return left_;
  }

  CosineNode* getRight() {
    return right_;
  }

 private:
  void CalStats();
  void ChooseCenter(Vector* center);
  void CalCosines(const Vector& center, ArrayList<double>* cosines);
  void CreateIndices(ArrayList<int>* indices);

  friend class CosineNodeTest;
  friend class CompareCosineNode;
};

class CompareCosineNode {
 public:
  bool operator ()(CosineNode* a, CosineNode* b) {
    return a->L2Err_ < b->L2Err_;
  }
};

#endif
