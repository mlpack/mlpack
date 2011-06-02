/** @file kde_dualtree.h
 *
 *  The template stub filled out for computing the kernel density
 *  estimate using a dual-tree algorithm.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_KDE_KDE_DUALTREE_H
#define MLPACK_KDE_KDE_DUALTREE_H

#include <boost/math/distributions/normal.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <deque>
#include "core/monte_carlo/mean_variance_pair.h"
#include "core/metric_kernels/kernel.h"
#include "core/tree/statistic.h"
#include "core/table/table.h"
#include "mlpack/series_expansion/hypercube_farfield_dev.h"
#include "mlpack/series_expansion/hypercube_local_dev.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "mlpack/series_expansion/multivariate_farfield_dev.h"
#include "mlpack/series_expansion/multivariate_local_dev.h"

// Sub-header files.
#include "mlpack/kde/kde_delta.h"
#include "mlpack/kde/kde_global.h"
#include "mlpack/kde/kde_postponed.h"
#include "mlpack/kde/kde_result.h"
#include "mlpack/kde/kde_summary.h"
#include "mlpack/kde/kde_statistic.h"

#endif
