
#ifndef SVM_MIN_QUAD_ON_SIMPLEX_H
#define SVM_MIN_QUAD_ON_SIMPLEX_H

#include <list>

namespace SVM_Projection {
  typedef std::list<index_t> SV_index;

  void OptimizeQuadraticOnSimplex(const Matrix& Q, Vector* x, SV_index* SVs);

  /** Project vector c on to canonical simplex,
   *  initialize vector x
   */
  void ProjectOnSimplex(const Vector& c, Vector* x, 
			SV_index* non_zeros);

  void OptimQuadratic(const Matrix& Q, Vector* x, SV_index* SVs);
};

#endif
