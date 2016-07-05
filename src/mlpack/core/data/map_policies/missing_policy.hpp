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
#include <limits>


using namespace std;

namespace mlpack {
namespace data {

/**
 * Same as increment map policy, but does not change type of features.
 */
class MissingPolicy
{
 public:
  // typedef of mapped_type
  using mapped_type = double;

  MissingPolicy()
  {
    // Nothing to initialize here.
  }

  explicit MissingPolicy(std::set<std::string> missingSet) :
    missingSet(std::move(missingSet))
  {
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
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (missingSet.count(string) != 0 &&
        (maps.count(dimension) == 0 ||
         maps[dimension].first.left.count(string) == 0))
    {
      // This string does not exist yet.
      size_t& numMappings = maps[dimension].second;

      typedef boost::bimap<std::string, mapped_type>::value_type PairType;
      maps[dimension].first.insert(PairType(string, NaN));

      ++numMappings;
      return NaN;
    }
    else
    {
      // This string already exists in the mapping or .
      return NaN;
    }
  }

  template <typename eT, typename MapType>
  void MapTokens(const std::vector<std::string>& tokens,
                 size_t& row,
                 arma::Mat<eT>& matrix,
                 MapType& maps,
                 std::vector<Datatype>& types)
  {
    std::stringstream token;
    for (size_t i = 0; i != tokens.size(); ++i)
    {
      token.str(tokens[i]);
      token>>matrix.at(row, i);
      if (token.fail()) // if not number, map it to datasetmapper
      {
        const eT val = static_cast<eT>(this->MapString(maps, types, tokens[i],
                                                       row));
        matrix.at(row, i) = val;
      }
      token.clear();
    }
  }

 private:
  std::set<std::string> missingSet;
}; // class MissingPolicy

} // namespace data
} // namespace mlpack

#endif
