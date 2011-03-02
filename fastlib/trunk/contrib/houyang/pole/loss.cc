// Implementation for Loss and derived classes

#include <cmath>

#include "loss.h"

//----------------------SquaredLoss---------------------------

SquaredLoss::SquaredLoss() {  
}
  
double SquaredLoss::GetLoss(double prediction, double label) {
  double example_loss = (prediction - label) * (prediction - label);
  return example_loss;
}

double SquaredLoss::GetUpdate(double prediction, double label) {
  return (label - prediction);
}


//----------------------HingeLoss---------------------------

HingeLoss::HingeLoss() {
}
  
double HingeLoss::GetLoss(double prediction, double label) {
  double e = 1 - label*prediction;
  return (e > 0) ? e : 0;
}
  
double HingeLoss::GetUpdate(double prediction, double label) {
  if ( (prediction * label) < 1.0)
    return label;
  else
    return 0.0;
}


//----------------------SquaredhingeLoss---------------------------

SquaredhingeLoss::SquaredhingeLoss() {
}
  
double SquaredhingeLoss::GetLoss(double prediction, double label) {
  double e = 1 - label*prediction;
  return (e > 0) ? 0.5*e*e : 0;
}
  
double SquaredhingeLoss::GetUpdate(double prediction, double label) {
  double e = 1 - label*prediction;
  return (e > 0) ? e : 0;
}


//----------------------LogisticLoss---------------------------

LogisticLoss::LogisticLoss() {
}
  
double LogisticLoss::GetLoss(double prediction, double label) {
  return log(1 + exp(-label * prediction));
}
  
double LogisticLoss::GetUpdate(double prediction, double label) {
  double d = exp(-label * prediction);
  return label * d / (1 + d);
}


//----------------------QuantileLoss---------------------------

QuantileLoss::QuantileLoss(double tau) : tau_(tau) {
}
  
double QuantileLoss::GetLoss(double prediction, double label) {
  double e = label - prediction;
  if(e > 0) {
    return tau_ * e;
  } else {
    return -(1 - tau_) * e;
  }
}
  
double QuantileLoss::GetUpdate(double prediction, double label) {
  double e = label - prediction;
  if(e == 0) return 0;
  if(e > 0) {
    return tau_;
  } else {
    return -(1 - tau_);
  }
}



