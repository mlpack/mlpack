/**
 * @file dataset_info_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the DatasetMapper<MapPolicy> class.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP

// In case it hasn't already been included.
#include "dataset_info.hpp"

namespace mlpack {
namespace data {

// Default constructor.
template<typename MapPolicy>
inline DatasetMapper<MapPolicy>::DatasetMapper(const size_t dimensionality) :
    types(dimensionality, Datatype::numeric)
{
  // Nothing to initialize.
}


// When we want to insert value into the map,
// we could use the policy to map the string
template<typename MapPolicy>
inline typename MapPolicy::mapped_type DatasetMapper<MapPolicy>::MapString(
                                                    const std::string& string,
                                                    const size_t dimension)
{
  return policy.template MapString<MapType>(maps, types, string, dimension);
}

// Return the string corresponding to a value in a given dimension.
template<typename MapPolicy>
inline const std::string& DatasetMapper<MapPolicy>::UnmapString(
                                                    const size_t value,
                                                    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.right.count(value) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<MapPolicy>::UnmapString(): value '" << value
        << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.right.at(value);
}

// Return the value corresponding to a string in a given dimension.
template<typename MapPolicy>
inline typename MapPolicy::mapped_type DatasetMapper<MapPolicy>::UnmapValue(
                                                    const std::string& string,
                                                    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.left.count(string) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetMapper<MapPolicy>::UnmapValue(): string '" << string
        << "' unknown for dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.left.at(string);
}

// Get the type of a particular dimension.
template<typename MapPolicy>
inline Datatype DatasetMapper<MapPolicy>::Type(const size_t dimension) const
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

template<typename MapPolicy>
inline Datatype& DatasetMapper<MapPolicy>::Type(const size_t dimension)
{
  if (dimension >= types.size())
    types.resize(dimension + 1, Datatype::numeric);

  return types[dimension];
}

template<typename MapPolicy>
inline
size_t DatasetMapper<MapPolicy>::NumMappings(const size_t dimension) const
{
  return (maps.count(dimension) == 0) ? 0 : maps.at(dimension).second;
}

template<typename MapPolicy>
inline size_t DatasetMapper<MapPolicy>::Dimensionality() const
{
  return types.size();
}

} // namespace data
} // namespace mlpack

#endif
