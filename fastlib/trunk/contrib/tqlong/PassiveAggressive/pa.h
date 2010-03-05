
#ifndef PASSIVE_AGGRESSIVE_H
#define PASSIVE_AGGRESSIVE_H

#include <fastlib/fastlib.h>

typedef double (*LossFunction)(const Vector& weight,const Vector& x,double y);

double hinge_loss(const Vector& weight,const Vector& x,double y) {
  double loss = 1 - la::Dot(weight, x) * y;
  return (loss > 0) ? loss : 0;
}

double LengthEuclideanSquare(const Vector& x) {
  double s = 0;
  for (index_t i = 0; i < x.length(); i++)
    s += x[i]*x[i];
  return s;
}

/** Implement a PA update on the ``weight'' 
    when seeing a sample and its label (x, y)
    - loss is a hinge loss
    - weight remains unchanged if no loss occur
    - otherwise, weight changes to nearest point that has zero loss on sample.
    Return: the loss
 */
double PA_Update(const Vector& w_t, const Vector& x_t, double y_t,
		 Vector& w_out) {
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / LengthEuclideanSquare(x_t);
  w_out.Copy(w_t);
  la::AddExpert(tau*y_t, x_t, &w_out);
  return loss_t;
}

double PA_Update(const Vector& w_t, double* x_t, double y_t,
		 Vector& w_out) {
  index_t n = w_t.length();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return PA_Update(w_t, X_t, y_t, w_out);
}

double PA_I_Update(const Vector& w_t, const Vector& x_t, double y_t,
		   double C, Vector& w_out) {
  double loss_t = hinge_loss(w_t, x_t, y_t);
  double tau = loss_t / LengthEuclideanSquare(x_t);
  if (tau > C) tau = C;
  w_out.Copy(w_t);
  la::AddExpert(tau*y_t, x_t, &w_out);
  return loss_t;
}

double PA_I_Update(const Vector& w_t, double* x_t, double y_t,
		   double C, Vector& w_out) {
  index_t n = w_t.length();
  Vector X_t; 
  X_t.Alias(x_t, n);
  return PA_I_Update(w_t, X_t, y_t, C, w_out);
}


#endif /* PASSIVE_AGGRESSIVE_H */
