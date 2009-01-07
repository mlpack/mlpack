#ifndef FUNCTION_H
#define FUNCTION_H

class Function: public Operator {

  /** @brief A function evaluation is evaluated exactly for naive and
   *         Monte Carlo style computation.
   */
  double MonteCarloCompute
  (const std::map<index_t, index_t> &constant_dataset_indices) {
    return NaiveCompute();
  }

}

#endif
