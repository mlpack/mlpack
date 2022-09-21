/**
 * @file core/metrics/metrics.hpp
 * @author Ryan Curtin
 *
 * Include all distance metrics implemented by mlpack.  Note that these are not
 * performance metrics for models---see core/cv/metrics/metrics.hpp instead.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_METRICS_HPP
#define MLPACK_CORE_METRICS_METRICS_HPP

#include "bleu.hpp" // Technically this should go somewhere else...
#include "iou_metric.hpp"
#include "ip_metric.hpp"
#include "lmetric.hpp"
#include "mahalanobis_distance.hpp"
#include "non_maximal_suppression.hpp"

#endif
