/**
 * @file size_checks.hpp
 * @author Kirill Mishchenko
 * @author Bisakh Mondal
 *
 * Utility for checking same size & same dimensionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_UTIL_SIZE_CHECKS_HPP
#define MLPACK_UTIL_SIZE_CHECKS_HPP


namespace mlpack {
namespace util {

/**
 * Check for if the given data points & labels have same size.
 *
 * @param data data.
 * @param labels Labels.
 * @param callerDescription A description of the caller that can be used for
 *     error generation.
 * @param addInfo Name to use for labels for precise error generation. Default
 *     is "labels"; for example, "weights" could also be used.
 * @param isDataTranspose Bool parameter which can be set true to transpose data
 *     before size-check. Default is false.
 * @param isLabelTranspose Bool parameter which can be set true to transpose label
 *     before size-check. Default is false.
 */
template<typename DataType, typename LabelsType>
inline void CheckSameSizes(
    const DataType& data,
    const LabelsType& label,
    const std::string& callerDescription,
    const std::string& addInfo = "labels",
    const bool& isDataTranspose = false,
    const bool& isLabelTranspose = false,
    const std::enable_if_t<!std::is_integral_v<LabelsType>>* = 0)
{
  const size_t dataPoints = (isDataTranspose == true) ? data.n_rows :
      data.n_cols;
  const size_t labelPoints = (isLabelTranspose == true) ? label.n_rows :
      label.n_cols;

  if (dataPoints != labelPoints)
  {
    std::ostringstream oss;
    oss << callerDescription << ": number of points (" << dataPoints << ") "
        << "does not match number of " << addInfo << " (" << labelPoints
        << ")!" << std::endl;
    throw std::invalid_argument(oss.str());
  }
}

/**
 * An overload of CheckSameSizes() where the size to be checked is known
 * previously. The second parameter is of type unsigned int.
 */
template<typename DataType, typename SizeType>
inline void CheckSameSizes(
    const DataType& data,
    const SizeType& size,
    const std::string& callerDescription,
    const std::string& addInfo = "labels",
    const std::enable_if_t<std::is_integral_v<SizeType>>* = 0)
{
  if (data.n_cols != size)
  {
    std::ostringstream oss;
    oss << callerDescription << ": number of points (" << data.n_cols << ") "
        << "does not match number of " << addInfo << " (" << size << ")!"
        << std::endl;
    throw std::invalid_argument(oss.str());
  }
}


/**
 * Check for if the given dataset dimension matches with the model's.
 *
 * @param data dataset.
 * @param dimension Dimension of the model.
 * @param callerDescription A description of the caller that can be used for
 *     error generation.
 * @param addInfo Name to use for dataset for precise error generation. Default
 *     is "dataset"; for example, "weights" could also be used.
 */
template<typename DataType, typename DimType>
inline void CheckSameDimensionality(
    const DataType& data,
    const DimType& dimension,
    const std::string& callerDescription,
    const std::string& addInfo = "dataset",
    const std::enable_if_t<!std::is_integral_v<DimType>>* = 0)
{
  if (data.n_rows != dimension.n_rows)
  {
    std::ostringstream oss;
    oss << callerDescription << ": dimensionality of " << addInfo << " ("
        << data.n_rows << ") is not equal to the dimensionality of the model"
        " (" << dimension.n_rows << ")!";

    throw std::invalid_argument(oss.str());
  }
}

/**
 * An overload of CheckSameDimensionality() where the dimension to be checked
 * is known second param is unsigned long int.
 */
template<typename DataType, typename DimType>
inline void CheckSameDimensionality(
    const DataType& data,
    const DimType& dimension,
    const std::string& callerDescription,
    const std::string& addInfo = "dataset",
    const std::enable_if_t<std::is_integral_v<DimType>>* = 0)
{
  if (data.n_rows != dimension)
  {
    std::ostringstream oss;
    oss << callerDescription << ": dimensionality of " << addInfo << " ("
        << data.n_rows << ") is not equal to the dimensionality of the model"
        " (" << dimension << ")!";
    throw std::invalid_argument(oss.str());
  }
}

} // namespace util
} // namespace mlpack

#endif
