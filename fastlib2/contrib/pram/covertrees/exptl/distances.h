#ifndef PARTIAL_DISTANCE_COMPUTATION_H
#define PARTIAL_DISTANCE_COMPUTATION_H

#include <fastlib/fastlib.h>
#define EPS 1.0e-3

namespace pdc {

  template<typename T>
  T DistanceEuclidean(GenVector<T>& x, GenVector<T>& y, T upper_bound) {

    DEBUG_SAME_SIZE(x.length(), y.length());
    T *vx = x.ptr();
    T *vy = y.ptr();
    T *end = vx + x.length();
    //index_t length = x.length();
    T s = 0.0;
    index_t batch = 120;
    //index_t t =(index_t) ((T) length / (T) batch);
    //index_t t1 = t;

    upper_bound *= upper_bound;

        
    //    while(t--) {
    //do {
    for (T *batch_end = vx + batch; batch_end <= end; batch_end += batch) {
      for (; vx != batch_end; vx +=2, vy += 2) {
	T d = *vx - *vy;
	//s += d * d;
	T d1 = *(vx+1) - *(vy+1);
	d *= d;
	d1 *= d1;
	s = s + d + d1;
	//	batch -= 2;
	//	vx += 2;
	//	vy += 2;
	// }while(batch);
      }
      if (s > upper_bound) {
	return sqrt(s) + EPS;
      }
      //      batch = 120;
    }
    

    //length = length - t1 * 120;

    //while (length) {
    for (; vx != end; /*vx += 2, vy += 2*/) {
      T d = *vx++ - *vy++;
      //s += d * d;
      //T d1 = *(vx+1) - *(vy+1);
      d *= d;
      //d1*= d1;
      //s = s + d + d1;
      s += d;
      // length -= 2;
      // vx += 2;
      // vy += 2;
    }
    
    return sqrt(s);
  }

};

#endif
