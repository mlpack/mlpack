#ifndef OPERATOR_H
#define OPERATOR_H

#include "fastlib/fastlib.h"

class Operator {

 private:

  /** @brief The nested operators under this operator.
   */
  ArrayList<Operator *> operators_;

  /** @brief The list of indices involved with this operator.
   */
  ArrayList<index_t> indices_;

  OT_DEF_BASIC(Operator) {
    OT_MY_OBJECT(operators_);
    OT_MY_OBJECT(indices_);
  }
  
 public:

  /** @brief Evaluate the operator exactly.
   */
  virtual double NaiveCompute() = 0;

  /** @brief Evaluate the operator using Monte Carlo.
   */
  virtual double MonteCarloCompute() = 0;

};

#endif
