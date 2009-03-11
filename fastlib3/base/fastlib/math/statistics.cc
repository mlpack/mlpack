// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file statistics.cc
 *
 * Implementation for statistics helpers.
 */

#include "statistics.h"
#include "math.h"

namespace math {

double Mean(Vector V) {
  double c = 0.0;
  index_t n = V.length();
  for (index_t i=0; i<n; i++)
    c = c + V[i];
  return c / n;
}

double Var(Vector V) {
  double c = 0.0, mean, va, ep;
  index_t n = V.length();

  for (index_t i=0; i<n; i++)
    c = c + V[i];
  mean = c / n;

  ep = 0.0; va = 0.0;
  for (index_t i=0; i<n; i++) {
    c = V[i] - mean;
    ep = ep + c;
    va = va + c * c;
  }
  return (va - ep * ep / n) / (n - 1);
}

double Std(Vector V) {
  return sqrt( Var(V) );
}
  
double Sigmoid(double x) {
  return 1.0 / ( 1.0 + exp(-x) );
}
  

};
