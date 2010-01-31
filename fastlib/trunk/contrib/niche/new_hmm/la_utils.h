#ifndef LA_UTILS_H
#define LA_UTILS_H

#include "fastlib/fastlib.h"

double Sum(const Vector &x) {
  double sum = 0;
  for(int i = 0; i < x.length(); i++) {
    sum += x[i];
  }
  return sum;
}

void HadamardMultiplyInit(const Vector &x,
			  const Vector &y,
			  Vector* z) {
  z -> Init(x.length());
  for(int i = 0; i < x.length(); i++) {
    (*z)[i] = x[i] * y[i];
  }
}

void HadamardMultiplyOverwrite(const Vector &x,
			       const Vector &y,
			       Vector* z) {
  for(int i = 0; i < x.length(); i++) {
    (*z)[i] = x[i] * y[i];
  }
}
 
void HadamardMultiplyBy(const Vector &x,
			Vector* y) {
  for(int i = 0; i < x.length(); i++) {
    (*y)[i] *= x[i];
  }
}





#endif /*LA_UTILS_H */
