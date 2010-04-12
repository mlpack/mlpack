/** 
 *  @file project_simplex.cc
 *
 *  This file implements the projection of a point on to
 *  the canonical simplex x_i >= 0 \sum x_i = 1
 * 
 *  @see project_simplex.h
 */
#include <fastlib/fastlib.h>
#include <list>
#include "svm_projection.h"

namespace SVM_Projection {

void ProjectOnSimplex(const Vector& c, Vector* x, 
		      SV_index* SVs) {
  Vector& _x = *x;
  SV_index& _SVs = *SVs;

  double s = 0;
  index_t n = c.length();
  index_t n_SV = c.length();

  _SVs.clear();
  _x.CopyValues(c);
  for (index_t i = 0; i < n; i++) {
    s += _x[i];
    _SVs.push_back(i);
  }

  bool done = false;
  double tol = 1e-6;
  while (!done) {
    double ds = 0;
    done = true;
    s = (s-1)/n_SV;
    SV_index::iterator it = _SVs.begin(); 
    do {
      int i = (*it);
      _x[i] -= s;
      if (_x[i] <= tol) {
	ds += _x[i];
	_x[i] = 0;
	n_SV--;
	it = _SVs.erase(it);
	done = false;
      }
      else it++;
    } while (it != _SVs.end());
    s = 1 - ds;
  }

}

};
