#ifndef FUNCTION_H
#define FUNCTION_H

#include "fastlib/fastlib.h"

class Function: public Operator {

  /** @brief A function evaluation is evaluated exactly for naive and
   *         Monte Carlo style computation.
   */
  double MonteCarloCompute
  (std::map<index_t, index_t> &constant_dataset_indices) {
    return NaiveCompute();
  }

}

#endif
