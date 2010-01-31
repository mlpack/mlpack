#ifndef LIN_ALGEBRA_CLASS_H_
#define LIN_ALGEBRA_CLASS_H_
#include "fastlib/fastlib.h"

class QuadraticObjective {
 public:
  void Init(Matrix &quadratiic_term, Vector &linear_term); //Initialize the member data of a class
  void Destruct(); //decinstructor
  void ComputeObjective(Vector &x, double *objective);
  void ComputeGradient(Vector &x, Vector *gradient);
  void ComputeHessian(Vector &x, Matrix *hessian);
 
private:
  //memeber variable with _(under score)
  Matrix quadratic_term_;
  Vector linear_term_;
  Matrix sum_quad_;
};



#endif
