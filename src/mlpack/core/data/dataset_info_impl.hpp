/**
 * @file dataset_info_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the DatasetMapper<PolicyType> class.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP

// In case it hasn't already been included.
#include "dataset_info.hpp"

namespace mlpack {
namespace data {

// Default constructor.
template<typename PolicyType>
inline DatasetMapper<PolicyType>::DatasetMapper(const size_t dimensionality) :
    types(dimensionality, Datatype::numeric)
{
    Log::Debug << "DatasetMapper(dimensionality)" << std::endl;
  // Nothing to initialize here.
}

template<typename PolicyType>
inline DatasetMapper<PolicyType>::DatasetMapper(PolicyType& policy,
                                               const size_t dimensionality) :
    types(dimensionality, Datatype::numeric),
    policy(std::move(policy))
{
    Log::Debug << "DatasetMapper(policy, dimensionality)" << std::endl;
  // Nothing to initialize here.
}

// When we want to insert value into the map,
// we could use the policy to map the string
template<typename PolicyType>
inline typename PolicyType::mapped_type DatasetMapper<PolicyType>::MapString(
                                                    const std::string& string,
                                                    const size_t dimension)
{
  return policy.template MapString<MapType>(maps, types, string, dimension);
}

// Return the string corresponding to a value in a given dimension.
template<typename PolicyType>
inline const std::string& DatasetMapper<PolicyType>::UnmapString(
                                                    const size_t value,
                                                    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.right.count(value) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<PolicyType>::UnmapString(): value '" << value
        << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.right.at(value);
}

// Return the value corresponding to a string in a given dimension.
template<typename PolicyType>
inline typename PolicyType::mapped_type DatasetMapper<PolicyType>::UnmapValue(
                                                    const std::string& string,
                                                    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.left.count(string) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<PolicyType>::UnmapValue(): string '" << string
        << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.left.at(string);
}

// Get the type of a particular dimension.
template<typename PolicyType>
inline Datatype DatasetMapper<PolicyType>::Type(const size_t dimension) const
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

template<typename PolicyType>
inline Datatype& DatasetMapper<PolicyType>::Type(const size_t dimension)
{
  if (dimension >= types.size())
    types.resize(dimension + 1, Datatype::numeric);

  return types[dimension];
}

template<typename PolicyType>
inline
size_t DatasetMapper<PolicyType>::NumMappings(const size_t dimension) const
{
  return (maps.count(dimension) == 0) ? 0 : maps.at(dimension).second;
}

template<typename PolicyType>
inline size_t DatasetMapper<PolicyType>::Dimensionality() const
{
  return types.size();
}

template<typename PolicyType>
inline PolicyType& DatasetMapper<PolicyType>::Policy() const
{
  return this->policy;
}

} // namespace data
} // namespace mlpack

#endif
