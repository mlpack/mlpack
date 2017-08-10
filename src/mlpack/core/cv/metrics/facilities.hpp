/**
 * @file facilities.hpp
 * @author Kirill Mishchenko
 *
 * Functionality that is used more than in one metric.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CV_METRICS_FACILITIES_HPP
#define MLPACK_CORE_CV_METRICS_FACILITIES_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace cv {

/**
  * Assert there is the same number of the given data points and labels.
  *
  * @param data Column-major data.
  * @param labels Labels.
  * @param callerDescription A description of the caller that can be used for
  *     error generation.
  */
template<typename DataType>
void AssertSizes(const DataType& data,
                 const arma::Row<size_t>& labels,
                 const std::string& callerDescription)
{
  if (data.n_cols != labels.n_elem)
  {
    std::ostringstream oss;
    oss << callerDescription << ": number of points (" << data.n_cols << ") "
        << "does not match number of labels (" << labels.n_elem << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
}

} // namespace cv
} // namespace mlpack

#endif
