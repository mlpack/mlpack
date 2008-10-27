#include "lin_algebra_class.h"

QuadraticObjective::Init(Matrix &quadratiic_term, Vector &linear_term) {
  DEBUG_SAME_SIZE(quadratic_term.n_rows(), quadratic_term.n_cols());
  DEBUG_SAME_SIZE(linear_term.length(), quadratic_term.n_rows());
  DEBUG_ASSERT_MSG(linear_term.length()==quadratic_term.n_rows(), 
      "Quadratic term and linear term don't have the same dimension %i != %i",
      quadratic_term.n_rows(), linear_term.lenght());
  quadratic_term_.Copy(quadratic_term);
  linear_term_.Copy(linear_term);
}

void QuadraticObjective::ComputeObjective(Vector &x, double *objective) {
  DEBUG_SAME_SIZE(x.length(), quadratic_term_.n_rows()); 
  Vector temp1;
  la::MulInit(x, quadratic_term_, &temp1);
  *objective = la::Dot(x, temp1) + la::Dot(linear_term, x);
}

