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
  // typedef of mapped_type
  using mapped_type = size_t;

  template <typename MapType>
  mapped_type MapString(const std::string& string,
                        const size_t dimension,
                        MapType& maps,
                        std::vector<Datatype>& types)
  {
    // If this condition is true, either we have no mapping for the given string
    // or we have no mappings for the given dimension at all.  In either case,
    // we create a mapping.
    if (maps.count(dimension) == 0 ||
        maps[dimension].first.left.count(string) == 0)
    {
      // This string does not exist yet.
      size_t& numMappings = maps[dimension].second;

      // change type of the feature to categorical
      if (numMappings == 0)
        types[dimension] = Datatype::categorical;

      typedef boost::bimap<std::string, mapped_type>::value_type PairType;
      maps[dimension].first.insert(PairType(string, numMappings));
      return numMappings++;
    }
    else
    {
      // This string already exists in the mapping.
      return maps[dimension].first.left.at(string);
    }
  }

  template <typename eT, typename MapType>
  void MapTokens(const std::vector<std::string>& tokens,
                 size_t& row,
                 arma::Mat<eT>& matrix,
                 MapType& maps,
                 std::vector<Datatype>& types)
  {
    auto notNumber = [](const std::string& str)
    {
      eT val(0);
      std::stringstream token;
      token.str(str);
      token >> val;
      return token.fail();
    };

    const bool notNumeric = std::any_of(std::begin(tokens),
                                        std::end(tokens), notNumber);
    if (notNumeric)
    {
       for (size_t i = 0; i != tokens.size(); ++i)
       {
         const eT val = static_cast<eT>(this->MapString(tokens[i], row, maps,
                                                        types));
         double temp = (double) val;
         matrix.at(row, i) = val;
       }
    }
    else
    {
      std::stringstream token;
      for (size_t i = 0; i != tokens.size(); ++i)
      {
         token.str(tokens[i]);
         token >> matrix.at(row, i);
         token.clear();
      }
    }
  }
}; // class IncrementPolicy

} // namespace data
} // namespace mlpack

#endif
