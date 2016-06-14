/**
 * @file increment_policy.hpp
 * @author Keon Kim
 *
 * Default increment maping policy for dataset info.
 */
#ifndef MLPACK_CORE_DATA_MAP_POLICIES_INCREMENT_POLICY_HPP
#define MLPACK_CORE_DATA_MAP_POLICIES_INCREMENT_POLICY_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>
#include <mlpack/core/data/map_policies/datatype.hpp>

using namespace std;

namespace mlpack {
namespace data {

/**
 * This class is used to map strings to incrementing unsigned integers (size_t).
 * First string to be mapped will be mapped to 0, next to 1 and so on.
 */
class IncrementPolicy
{
 public:
  typedef size_t mapped_type;

  template <typename MapType>
  mapped_type MapString(MapType& maps,
                       std::vector<Datatype>& types,
                       const std::string& string,
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
}; // class IncrementPolicy

} // namespace data
} // namespace mlpack

#endif
