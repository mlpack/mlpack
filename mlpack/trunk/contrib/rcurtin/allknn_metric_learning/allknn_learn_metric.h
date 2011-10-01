/***
 * @file allknn_learn_metric.h
 * @author Ryan Curtin
 *
 * Our objective is to learn a metric to optimize the performance of
 * all-k-nearest-neighbors.  The specific thing we are seeking is the matrix A
 * in the distance calculation d(x_i, x_j) = x_i^T * A * x_j; in this method we
 * have restricted A to being diagonal.
 *
 * A user passes in the dataset which is to be analyzed with AllkNN, an output
 * file to store the weights, and optionally a file of initial weights to start
 * from (if, for instance, a better set of weights is already known).  The
 * number of iterations performed, the number of neighbors k, and the learning
 * rate alpha can all be specified.
 *
 * This code uses the old-style FASTLIB FX system (which will need to be
 * changed).
 */

#ifndef __MLPACK_CONTRIB_RCURTIN_ALLKNN_LEARN_METRIC_H
#define __MLPACK_CONTRIB_RCURTIN_ALLKNN_LEARN_METRIC_H

#include <fastlib/fastlib.h>
#include <mlpack/allknn/allknn.h>

#include "allknn_metric_utils.h"

// Define options
const fx_entry_doc allnn_timit_main[] = {
  {"input_file", FX_REQUIRED, FX_STR, NULL, "Input CSV file."},
  {"output_file", FX_REQUIRED, FX_STR, NULL,
      "Output CSV file; weights are exported."},
  {"k", FX_REQUIRED, FX_INT, NULL, "k-nearest-neighbors value."},
  {"alpha", FX_PARAM, FX_DOUBLE, NULL, "Learning rate (default 0.7)."},
  {"beta", FX_PARAM, FX_DOUBLE, NULL, "Learning slowdown rate (default 0.95)."},
  {"max_iterations", FX_PARAM, FX_INT, NULL,
      "Maximum iterations to find perturbation improvement (default 100000)."},
  {"input_weights", FX_PARAM, FX_STR, NULL,
      "File containing weights to start with (default is [1 ... 1]^T)"},
  {"initial_step", FX_PARAM, FX_DOUBLE, NULL,
      "Initial step size (default 1.0)"},
  {"perturbation", FX_PARAM, FX_DOUBLE, NULL,
      "Size of perturbations to use in gradient estimates (default 0.05)"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc allnn_timit_doc = {
  allnn_timit_main, NULL, "AllkNN Perturbation-Based Metric Learning"
};

#endif
