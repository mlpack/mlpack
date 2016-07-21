/**
 * @file dataset_info_impl.hpp
 * @author Ryan Curtin
 *
 * An implementation of the DatasetInfo class.
 *
 * This file is part of mlpack 2.0.3.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_IMPL_HPP

// In case it hasn't already been included.
#include "dataset_info.hpp"

namespace mlpack {
namespace data {

// Default constructor.
inline DatasetInfo::DatasetInfo(const size_t dimensionality) :
    types(dimensionality, Datatype::numeric)
{
  // Nothing to initialize.
}

// Map the string to a numeric id.
inline size_t DatasetInfo::MapString(const std::string& string,
                                     const size_t dimension)
{
  // If this condition is true, either we have no mapping for the given string
  // or we have no mappings for the given dimension at all.  In either case,
  // we create a mapping.
  if (maps.count(dimension) == 0 ||
      maps[dimension].first.left.count(string) == 0)
  {
    // This string does not exist yet.
    size_t& numMappings = maps[dimension].second;
    if (numMappings == 0)
      types[dimension] = Datatype::categorical;
    typedef boost::bimap<std::string, size_t>::value_type PairType;
    maps[dimension].first.insert(PairType(string, numMappings));
    return numMappings++;
  }
  else
  {
    // This string already exists in the mapping.
    return maps[dimension].first.left.at(string);
  }
}

// Return the string corresponding to a value in a given dimension.
inline const std::string& DatasetInfo::UnmapString(
    const size_t value,
    const size_t dimension)
{
  // Throw an exception if the value doesn't exist.
  if (maps[dimension].first.right.count(value) == 0)
  {
    std::ostringstream oss;
    oss << "DatasetInfo::UnmapString(): value '" << value << "' unknown for "
        << "dimension " << dimension;
    throw std::invalid_argument(oss.str());
  }

  return maps[dimension].first.right.at(value);
}

// Get the type of a particular dimension.
inline Datatype DatasetInfo::Type(const size_t dimension) const
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

inline Datatype& DatasetInfo::Type(const size_t dimension)
{
  if (dimension >= types.size())
    types.resize(dimension + 1, Datatype::numeric);

  return types[dimension];
}

inline size_t DatasetInfo::NumMappings(const size_t dimension) const
{
  return (maps.count(dimension) == 0) ? 0 : maps.at(dimension).second;
}

inline size_t DatasetInfo::Dimensionality() const
{
  return types.size();
}

} // namespace data
} // namespace mlpack

#endif
