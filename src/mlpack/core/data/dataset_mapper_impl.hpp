/**
 * @file core/data/dataset_mapper_impl.hpp
 * @author Ryan Curtin
 * @author Keon Kim
 *
 * An implementation of the DatasetMapper<PolicyType> class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP

// In case it hasn't already been included.
#include "dataset_mapper.hpp"

namespace mlpack {
namespace data {

// Default constructor.
template<typename PolicyType, typename InputType>
inline DatasetMapper<PolicyType, InputType>::DatasetMapper(
    const size_t dimensionality) :
    types(dimensionality, Datatype::numeric)
{
  // Nothing to initialize here.
}

template<typename PolicyType, typename InputType>
inline DatasetMapper<PolicyType, InputType>::DatasetMapper(PolicyType& policy,
    const size_t dimensionality) :
    types(dimensionality, Datatype::numeric),
    policy(std::move(policy))
{
  // Nothing to initialize here.
}

template<typename PolicyType, typename InputType>
inline void DatasetMapper<PolicyType, InputType>::SetDimensionality(
    const size_t dimensionality)
{
  types = std::vector<Datatype>(dimensionality, Datatype::numeric);
  maps.clear();
}

// Utility helper function to call MapFirstPass.
template<typename PolicyType, typename InputType, typename T>
void CallMapFirstPass(
    PolicyType& policy,
    const InputType& input,
    const size_t dimension,
    std::vector<Datatype>& types,
    const std::enable_if_t<PolicyType::NeedsFirstPass>* = 0)
{
  policy.template MapFirstPass<T>(input, dimension, types);
}

// Utility helper function that doesn't call anything.
template<typename PolicyType, typename InputType, typename T>
void CallMapFirstPass(
    PolicyType& /* policy */,
    const InputType& /* input */,
    const size_t /* dimension */,
    std::vector<Datatype>& /* types */,
    const std::enable_if_t<!PolicyType::NeedsFirstPass>* = 0)
{
  // Nothing to do here.
}

template<typename PolicyType, typename InputType>
template<typename T>
void DatasetMapper<PolicyType, InputType>::MapFirstPass(const InputType& input,
                                                        const size_t dimension)
{
  // Call the correct overload (via SFINAE).
  CallMapFirstPass<PolicyType, InputType, T>(policy, input, dimension, types);
}

// When we want to insert value into the map, we use the policy to map the
// input.
template<typename PolicyType, typename InputType>
template<typename T>
inline T DatasetMapper<PolicyType, InputType>::MapString(
    const InputType& input,
    const size_t dimension)
{
  return policy.template MapString<MapType, T>(input, dimension, maps, types);
}

/**
 * A safe version of isnan() that only gets called when the type has a NaN at
 * all.  This is a workaround for Visual Studio, which doesn't seem to support
 * isnan(size_t).
 */
template<typename T>
inline bool isnanSafe(const T& /* t */)
{
  return false;
}

template<>
inline bool isnanSafe(const double& t)
{
  return std::isnan(t);
}

template<>
inline bool isnanSafe(const float& t)
{
  return std::isnan(t);
}

template<>
inline bool isnanSafe(const long double& t)
{
  return std::isnan(t);
}


// Return the input corresponding to a value in a given dimension.
template<typename PolicyType, typename InputType>
template<typename T>
inline const InputType& DatasetMapper<PolicyType, InputType>::UnmapString(
    const T value,
    const size_t dimension,
    const size_t unmappingIndex) const
{
  // If the value is std::numeric_limits<T>::quiet_NaN(), we can't use it as a
  // key---so we will use something else...
  const T usedValue = isnanSafe(value) ?
      std::nexttoward(std::numeric_limits<T>::max(), T(0)) :
      value;

  // Throw an exception if the value doesn't exist.
  if (maps.at(dimension).second.count(usedValue) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<PolicyType, InputType>::UnmapString(): value '"
        << value << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  if (unmappingIndex >= maps.at(dimension).second.at(usedValue).size())
  {
    std::ostringstream oss;
    oss << "DatasetMapper<PolicyType, InputType>::UnmapString(): value '"
        << value << "' only has "
        << maps.at(dimension).second.at(usedValue).size()
        << " unmappings, but unmappingIndex is " << unmappingIndex << "!";
    throw std::invalid_argument(oss.str());
  }

  return maps.at(dimension).second.at(usedValue)[unmappingIndex];
}

template<typename PolicyType, typename InputType>
template<typename T>
inline size_t DatasetMapper<PolicyType, InputType>::NumUnmappings(
    const T value,
    const size_t dimension) const
{
  // If the value is std::numeric_limits<T>::quiet_NaN(), we can't use it as a
  // key---so we will use something else...
  if (isnanSafe(value))
  {
    const T newValue = std::nexttoward(std::numeric_limits<T>::max(), T(0));
    return maps.at(dimension).second.at(newValue).size();
  }

  return maps.at(dimension).second.at(value).size();
}

// Return the value corresponding to an input in a given dimension.
template<typename PolicyType, typename InputType>
inline typename PolicyType::MappedType
DatasetMapper<PolicyType, InputType>::UnmapValue(
    const InputType& input,
    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.count(input) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<PolicyType, InputType>::UnmapValue(): input '"
        << input << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.at(input);
}

// Get the type of a particular dimension.
template<typename PolicyType, typename InputType>
inline Datatype DatasetMapper<PolicyType, InputType>::Type(
    const size_t dimension) const
{
  if (dimension >= types.size())
  {
    std::ostringstream oss;
    oss << "requested type of dimension " << dimension << ", but dataset only "
        << "has " << types.size() << " dimensions";
    throw std::invalid_argument(oss.str());
  }

  return types[dimension];
}

template<typename PolicyType, typename InputType>
inline Datatype& DatasetMapper<PolicyType, InputType>::Type(
    const size_t dimension)
{
  if (dimension >= types.size())
    types.resize(dimension + 1, Datatype::numeric);

  return types[dimension];
}

template<typename PolicyType, typename InputType>
inline size_t
DatasetMapper<PolicyType, InputType>::NumMappings(const size_t dimension) const
{
  return (maps.count(dimension) == 0) ? 0 : maps.at(dimension).first.size();
}

template<typename PolicyType, typename InputType>
inline size_t DatasetMapper<PolicyType, InputType>::Dimensionality() const
{
  return types.size();
}

template<typename PolicyType, typename InputType>
inline const PolicyType& DatasetMapper<PolicyType, InputType>::Policy() const
{
  return this->policy;
}

template<typename PolicyType, typename InputType>
inline PolicyType& DatasetMapper<PolicyType, InputType>::Policy()
{
  return this->policy;
}

template<typename PolicyType, typename InputType>
inline void DatasetMapper<PolicyType, InputType>::Policy(PolicyType&& policy)
{
  this->policy = std::forward<PolicyType>(policy);
}

} // namespace data
} // namespace mlpack

#endif
