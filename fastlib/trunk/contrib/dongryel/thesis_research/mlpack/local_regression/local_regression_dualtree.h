/** @file local_regression_dualtree.h
 *
 *  The template stub filled out for computing the local regression
 *  estimate using a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DUALTREE_H
#define MLPACK_LOCAL_REGRESSION_LOCAL_REGRESSION_DUALTREE_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi.hpp>
#include <boost/scoped_array.hpp>
#include <boost/serialization/serialization.hpp>
#include <deque>
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/monte_carlo/mean_variance_pair_matrix.h"
#include "core/metric_kernels/kernel.h"
#include "core/tree/statistic.h"
#include "core/table/table.h"
#include "mlpack/local_regression/local_regression_delta.h"
#include "mlpack/local_regression/local_regression_global.h"
#include "mlpack/local_regression/local_regression_postponed.h"
#include "mlpack/local_regression/local_regression_result.h"
#include "mlpack/local_regression/local_regression_summary.h"
#include "mlpack/local_regression/local_regression_statistic.h"

#endif
