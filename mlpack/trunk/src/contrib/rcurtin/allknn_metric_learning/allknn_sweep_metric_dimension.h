/***
 * @file allknn_sweep_metric_dimension.h
 * @author Ryan Curtin
 *
 * This utility sweeps a single dimension of an AllkNN metric and performs the
 * computation for each value in the specified range of values.  So, for
 * instance, we could evaluate AllkNN on a 30-dimensional dataset where we are
 * testing all weights on dimension 12 from 0:0.02:5 (we can also specify step
 * size).
 */

#ifndef __MLPACK_CONTRIB_RCURTIN_ALLKNN_SWEEP_METRIC_DIMENSCLIN_H
#define __MLPACK_CONTRIB_RCURTIN_ALLKNN_SWEEP_METRIC_DIMENSCLIN_H

#include <fastlib/fastlib.h>
#include <mlpack/allknn/allknn.h>

#include "allknn_metric_utils.h"

// Define options
const fx_entry_doc allnn_timit_main[] = {
  {"input_file", FX_REQUIRED, FX_STR, NULL, "Input CSV file."},
  {"k", FX_PARAM, FX_INT, NULL, "k-nearest-neighbors value."},
  {"dim", FX_PARAM, FX_INT, NULL, "Dimension to sweep over."},
  {"step", FX_PARAM, FX_DOUBLE, NULL, "Step size to use."},
  {"spread", FX_PARAM, FX_DOUBLE, NULL, "Range to optimize over [1 - spread, 1 + spread]"},
  FX_ENTRY_DOC_DONE
};

const fx_module_doc allnn_timit_doc = {
  allnn_timit_main, NULL, "AllkNN Metric Dimension Sweep Tool"
};

#endif
