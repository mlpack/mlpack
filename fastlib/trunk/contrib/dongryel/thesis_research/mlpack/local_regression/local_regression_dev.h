/** @file local_regression_dev.h
 *
 *  The implementation of mixed logit discrete choice model.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DEV_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DEV_H

#include "mlpack/local_regression/local_regression.h"

namespace mlpack {
namespace local_regression {

template<typename TableType, typename KernelType>
void LocalRegression<TableType, KernelType>::Init(
  ArgumentType &arguments_in) {

  // Initialize the table for storing/accessing the attribute vector
  // for each person.
  table_.Init(arguments_in);
}

template<typename TableType, typename KernelType>
void LocalRegression<TableType, KernelType>::Compute(
  const ArgumentType &arguments_in,
  mlpack::local_regression::LocalRegressionResult *result_out) {

}

}
}

#endif
