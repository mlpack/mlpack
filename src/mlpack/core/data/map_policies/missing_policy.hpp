/**
 * @file missing_policy.hpp
 * @author Keon Kim
 *
 * Missing map policy for dataset info.
 */
#ifndef MLPACK_CORE_DATA_MAP_POLICIES_MISSING_POLICY_HPP
#define MLPACK_CORE_DATA_MAP_POLICIES_MISSING_POLICY_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>
#include <mlpack/core/data/map_policies/datatype.hpp>


using namespace std;

namespace mlpack {
namespace data {

/**
 * Same as increment map policy, but does not change type of features.
 */
class MissingPolicy
{
 public:
  typedef size_t mapped_type;

  MissingPolicy()
  {
    Log::Debug << "MissingPolicy()" << std::endl;
    missingSet.insert("a");
  }

  explicit MissingPolicy(std::set<std::string> missingSet) :
    missingSet(std::move(missingSet))
  {
    Log::Debug << "MissingPolicy()" << std::endl;
    // Nothing to initialize here.
  }


  template <typename MapType>
  mapped_type MapString(MapType& maps,
                        std::vector<Datatype>& types,
                        const std::string& string,
                        const size_t dimension)
  {
    // If this condition is true, either we have no mapping for the given string
    // or we have no mappings for the given dimension at all.  In either case,
    // we create a mapping.
    if (missingSet.count(string) != 0 &&
        maps.count(dimension) == 0 ||
        maps[dimension].first.left.count(string) == 0)
    {
      // This string does not exist yet.
      size_t& numMappings = maps[dimension].second;

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
 private:
  std::set<std::string> missingSet;
}; // class MissingPolicy

} // namespace data
} // namespace mlpack

#endif
