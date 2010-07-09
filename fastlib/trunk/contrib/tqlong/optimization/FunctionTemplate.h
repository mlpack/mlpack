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
  * 0-order smooth function: CalculateValue
  * 1-order smooth function: CalculateValue, CalculateGradient
  * 2-order smooth function: CalculateValue, CalculateGradient, CalculateHessian
  *************************************************************/

class LengthEuclidianSquare {
public:
  double CalculateValue(const Vector& x);
  void CalculateGradient(const Vector& x, Vector& gradient);
  void CalculateHessian(const Vector& x, Matrix& hessian);
};

END_OPTIM_NAMESPACE;

#endif // FUNCTIONTEMPLATE_H
