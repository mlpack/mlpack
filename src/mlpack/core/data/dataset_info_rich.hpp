/**
 * @file dataset_info.hpp
 * @author Ryan Curtin
 *
 * Defines the DatasetInfo class, which holds information about a dataset.  This
 * is useful when the dataset contains categorical non-numeric features that
 * needs to be mapped to categorical numeric features.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_RICH_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_RICH_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include "map_policies/default_map_policy.hpp"
#include <boost/bimap.hpp>

namespace mlpack {
namespace data {

template<typename MapPolicy>
class DatasetInfoRich
{
 public:

  DatasetInfoRich(const size_t dimensionality = 0):
      types(dimensionality, Datatype::numeric)
  {
    // nothing to initialize
  }

  typename MapPolicy::map_type_t MapString(const std::string& string,
                                           const size_t dimension)
  {
    return policy.template MapString<MapType>(maps, string, dimension);
  }


  // Return the value corresponding to a string in a given dimension.
  typename MapPolicy::map_type_t UnmapValue(const std::string& string,
                                            const size_t dimension) const
  {
    return 0;
  }

  size_t NumMappings(const size_t dimension) const
  {
    return (maps.count(dimension) == 0) ? 0 : maps.at(dimension).second;
  }

 private:

  //! Types of each dimension.
  std::vector<Datatype> types;

  //! Mappings from strings to integers.  Map entries will only exist for
  //! dimensions that are categorical.
  typedef std::unordered_map<size_t,
      std::pair<boost::bimap<std::string, typename MapPolicy::map_type_t>,
                size_t>> MapType;

  MapType maps;
  //using PairType =
      //boost::bimap<std::string, typename MapPolicy::map_type_t>::value_type;

  MapPolicy policy;
};

using DefaultDatasetInfo = DatasetInfoRich<data::DefaultMapPolicy>;

} // namespace data
} // namespace mlpack

#endif
