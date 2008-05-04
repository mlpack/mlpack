#ifndef PARTIAL_DISTANCE_COMPUTATION_H
#define PARTIAL_DISTANCE_COMPUTATION_H

#include <fastlib/fastlib.h>
#define EPS 1.0e-5

namespace pdc {

  double DistanceEuclidean(Vector& x, Vector& y, double upper_bound) {

    DEBUG_SAME_SIZE(x.length(), y.length());
    double *vx = x.ptr();
    double *vy = y.ptr();
    index_t length = x.length();
    double s = 0;
    index_t batch = 120;
    index_t t =(index_t) ((double) length / (double) batch);
    index_t t1 = t;

    upper_bound *= upper_bound;

    do {
      do {
	double d = *vx++ - *vy++;
	s += d * d;
	d = *vx++ - *vy++;
	s += d * d;
	//d = *vx++ - *vy++;
	//s += d * d;
	//d = *vx++ - *vy++;
	//s += d * d;
	batch -= 2;
      }while(batch);
      if (s > upper_bound) {
	return sqrt(s) + EPS;
      }
      batch = 120;
    }while(--t);

    length = length - t1 * 120;

    while (length) {
      double d = *vx++ - *vy++;
      s += d * d;
      d = *vx++ - *vy++;
      s += d * d;
      length -= 2;
    }

    if (s > upper_bound) {
      return sqrt(s) + EPS;
    }
    else {
      return sqrt(s);
    }

  }

};

#endif
