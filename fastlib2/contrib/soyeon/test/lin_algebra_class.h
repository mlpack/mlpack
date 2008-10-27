#ifndef LIN_ALGEBRA_CLASS_H_
#define LIN_ALGEBRA_CLASS_H_
#include "fastlib/fastlib.h"

class QuadraticObjective {
 public:
  void Init(Matrix &quadratiic_term, Vector &linear_term); 
  void Destruct();
  void ComputeObjective(Vector &x, double *objective);
  void ComputeGradient(Vector &x, Vector *gradient);
  void ComputeHessian(Vector &x, Matrix *hessian);
 private:
  Matrix quadratic_term_;
  Vector linear_term_; 
};



#endif
