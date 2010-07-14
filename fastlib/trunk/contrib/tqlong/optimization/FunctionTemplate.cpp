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
  la::ScaleOverwrite(2.0, x, &gradient);
}

void optim::LengthEuclidianSquare::CalculateHessian(const Vector &x, Matrix &hessian)
{
  DEBUG_ASSERT(x.length() == hessian.n_rows() && x.length() == hessian.n_cols() && x.length() == dim);
  hessian.SetAll(0.0);
  for (int i = 0; i < hessian.n_rows(); i++)
    hessian.ref(i, i) = 2.0;
}

double optim::TestFunction::CalculateValue(const Vector &x)
{
  DEBUG_ASSERT(x.length() == 2);
  return pow(x[0]-2, 4) + pow(x[0]-2,2)*pow(x[1],2) + pow(x[1]+1,2);
}

void optim::TestFunction::CalculateGradient(const Vector &x, Vector &gradient)
{
  DEBUG_ASSERT(x.length() == 2 && gradient.length() == 2);
  gradient[0] = 4*pow(x[0]-2,3)+2*(x[0]-2)*pow(x[1],2);
  gradient[1] = pow(x[0]-2,2)*2*x[1]+2*(x[1]+1);
}

