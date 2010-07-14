#include <fastlib/fastlib.h>

#include "FunctionTemplate.h"

double optim::LengthEuclidianSquare::CalculateValue(const Vector &x)
{
  DEBUG_ASSERT(x.length() == dim);
  double s = 0;
  for (int i = 0; i < x.length(); i++)
    s += x[i]*x[i];
  return s;
}

void optim::LengthEuclidianSquare::CalculateGradient(const Vector &x, Vector &gradient)
{
  DEBUG_ASSERT(x.length() == dim && gradient.length() == dim);
  DEBUG_ASSERT(x.length() == gradient.length());
  la::ScaleOverwrite(2.0, x, &gradient);
}

void optim::LengthEuclidianSquare::CalculateHessian(const Vector &x, Matrix &hessian)
{
  DEBUG_ASSERT(x.length() == hessian.n_rows() && x.length() == hessian.n_cols() && x.length() == dim);
  hessian.SetAll(0.0);
  for (int i = 0; i < hessian.n_rows(); i++)
    hessian.ref(i, i) = 2.0;
}
