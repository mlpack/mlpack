/**
 * @file facilities.hpp
 * @author Kirill Mishchenko
 * @author Bisakh Mondal
 *
 * Utility that is used for checking same size &  same dimensionality between
 * data & response.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_UTIL_FACILITIES_HPP
#define MLPACK_UTIL_FACILITIES_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace util {

/**
  * Check for if the given data points & labels have same size.
  *
  * @param data data.
  * @param labels Labels.
  * @param callerDescription A description of the caller that can be used for
  *     error generation.
  * @param mode For nature of comparision(default "CE").
  *   types of mode:
  *     "CE" equivalent to (data.n_cols, labels.n_elem).
  *     "CC" equivalent to (data.n_cols, labels.n_cols).
  */
template<typename DataType, typename LabelsType>
inline void CheckSameSizes(const DataType& data,
                          const LabelsType& labels,
                          const std::string& callerDescription,
                          const std::string& mode = "CE")
{
  if (mode == "CE")
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
  else if (mode == "CC")
  {
    if (data.n_cols != labels.n_cols)
    {
      std::ostringstream oss;
      oss << callerDescription << ": number of points (" << data.n_cols << ") "
          << "does not match number of responses (" << labels.n_cols << ")!"
          << std::endl;
      throw std::invalid_argument(oss.str());
    }
  }
  else
    //For development purpose, not intended for user.
    Log::Fatal << "Ensure Providing Correct mode!!" << std::endl;
 
}

} // namespace util
} // namespace mlpack

#endif
