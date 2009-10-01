
#ifndef TQLONG_KFS_H
#define TQLONG_KFS_H

template <class DATA>
class FeatureKernel {
  DATA& X;
 public:
  FeatureKernel(DATA& X_) : X(X_) {}
  /** Compute kernel value between sample i and sample j at feature k */ 
  double getKernel(int i, int j, int k) {
    return DATA::getFeatureKernel(i, j, k);
  }
};

class MatrixData {
  Matrix& X;
 public:
  MatrixData(Matrix& X_) : X(X_) {}

  double getFeatureKernel(int i, int j, int k) {
    return X.get(k, i)*X.get(k, j); // linear kernel
  }
};

template <class DATA, class LABEL>
void KernelFeatureSelection(const LABEL& y, FeatureKernel<DATA>& kernel) {
  
}

#endif
