#ifndef FUNCTIONTEMPLATE_H
#define FUNCTIONTEMPLATE_H

#include <fastlib/fastlib.h>

#ifndef BEGIN_OPTIM_NAMESPACE
#define BEGIN_OPTIM_NAMESPACE namespace optim {
#endif
#ifndef END_OPTIM_NAMESPACE
#define END_OPTIM_NAMESPACE }
#endif

BEGIN_OPTIM_NAMESPACE;

/*************************************************************
  * A function template, implement CalculateXXXXX methods
  * Function value         : CalculateValue
  * 1-order smooth function: CalculateValue, CalculateGradient
  * 2-order smooth function: CalculateValue, CalculateGradient, CalculateHessian
  * Required:
      int dimension()
      void Init(variable_type*)
      variable_type (typedef)
  *************************************************************/

class LengthEuclidianSquare {
  int dim;
public:
  typedef Vector variable_type;                   // required
  int dimension() { return dim; }                 // required
  void Init(Vector* x) { x->Init(dim); }          // required
  double CalculateValue(const Vector& x);
  void CalculateGradient(const Vector& x, Vector& gradient);
  void CalculateHessian(const Vector& x, Matrix& hessian);
public:
  LengthEuclidianSquare(int dim_ = 2) : dim(dim_) {}
};

END_OPTIM_NAMESPACE;

#endif // FUNCTIONTEMPLATE_H
