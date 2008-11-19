#ifndef QUICSVD_QUICSVD_H
#define QUICSVD_QUICSVD_H
#include <fastlib/fastlib.h>
#include "cosine_tree.h"
#include <vector>
#include <queue>

class QuicSVD {
  typedef std::priority_queue<CosineNode*, std::vector<CosineNode*>,
    CompareCosineNode> CosineNodeQueue;

  Matrix A_;
  double dataNorm2_;

  CosineNode root_;
  CosineNodeQueue leaves_;

  ArrayList<Vector> basis_;
  ArrayList<Vector> UTA_;
  
  ArrayList<double> projMagSq_;
  double sumProjMagSq_;

  double targetRelErr_;

 public:
  QuicSVD(const Matrix& A, double targetRelErr);
  
  index_t n_cols() {
    return A_.n_cols();
  }

  void ComputeSVD(Vector* s, Matrix* U, Matrix* VT);

 private:
  void addBasisFrom(const CosineNode& node);
  double curRelErr();
  void addToQueue(CosineNode* node);
  double calL2Err(const CosineNode& node);
  void extractSVD(Vector* s, Matrix* U, Matrix* VT);

  friend class QuicSVDTest;
};
#endif
