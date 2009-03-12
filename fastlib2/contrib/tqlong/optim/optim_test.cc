
#include <fastlib/fastlib.h>
#include "optim.h"
#include "myfun.h"

/*
double my_fun(const Vector& x) { // sum x[i]^2
  for (int i = 0; i < x.length(); i++)
    if (x[i] < 0.5) return INFINITY;
  return la::Dot(x, x);
}

void my_grad(const Vector& x, Vector* g) { // g = 2x
  la::ScaleOverwrite(2, x, g);
}

double my_fun1(const Vector& x) { // sum x[i]^2
  Vector one;
  one.Init(x.length());
  one.SetAll(1.0);
  Vector xsub1;
  la::SubInit(one, x, &xsub1);
  return la::Dot(xsub1, xsub1);
}

void my_grad1(const Vector& x, Vector* g) { // g = 2x
  Vector one;
  one.Init(x.length());
  one.SetAll(1.0);
  Vector xsub1;
  la::SubInit(one, x, &xsub1);
  la::ScaleOverwrite(2, xsub1, g);
}

void AddExpert(double alpha, const Vector& x, Vector* y) {
  la::AddExpert(alpha, x, y);
}

double Dot(const Vector& x, const Vector& y) {
  return la::Dot(x, y);
}
*/
int main(int argc, char** argv) {
  Vector x, xnew;
  x.Init(2);
  xnew.Init(2);

  x[0] = 3; x[1] = 2;
  MyFun1 mf(2);

  //double f = optim::GradientDescent(mf, x, &xnew, 100, 1e-6, 1e-6);
  //double f = optim::BFGSDescent(mf, x, &xnew, 100, 1e-6, 1e-6);
  //double f = optim::L_BFGSDescent(mf, x, &xnew, 100, 1e-6, 1e-6,4);
  double f = optim::BarrierMethod(mf, x, &xnew, 100, 1e-6, 1e-6);

  ot::Print(xnew);
  ot::Print(f);
}
