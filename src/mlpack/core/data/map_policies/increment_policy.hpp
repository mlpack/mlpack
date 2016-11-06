/**
 * @file increment_policy.hpp
 * @author Keon Kim
 *
 * Default increment maping policy for dataset info.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MAP_POLICIES_INCREMENT_POLICY_HPP
#define MLPACK_CORE_DATA_MAP_POLICIES_INCREMENT_POLICY_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>
#include <mlpack/core/data/map_policies/datatype.hpp>

namespace mlpack {
namespace data {
/**
 * IncrementPolicy is used as a helper class for DatasetMapper. It tells how the
 * strings should be mapped. Purpose of this policy is to map all dimension if
 * one if the variables in a dimension turns out to be a categorical variable.
 * IncrementPolicy maps strings to incrementing unsigned integers (size_t).
 * The first string to be mapped will be mapped to 0, the next to 1 and so on.
 */
class IncrementPolicy
{
 public:
  // typedef of MappedType
  using MappedType = size_t;

  /**
   * Given the string and the dimension to which the it belongs, and the maps
   * and types given by the DatasetMapper class, returns its numeric mapping.
   * If no mapping yet exists, the string is added to the list of mappings for
   * the given dimension. This function is used as a helper function for
   * DatasetMapper class.
   *
   * @tparam MapType Type of unordered_map that contains mapped value pairs
   * @param string String to find/create mapping for.
   * @param dimension Index of the dimension of the string.
   * @param maps Unordered map given by the DatasetMapper.
   * @param types Vector containing the type information about each dimensions.
   */
  template <typename MapType>
  MappedType MapString(const std::string& string,
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

      typedef boost::bimap<std::string, MappedType>::value_type PairType;
      maps[dimension].first.insert(PairType(string, numMappings));
      return numMappings++;
    }
    else
    {
      // This string already exists in the mapping.
      return maps[dimension].first.left.at(string);
    }
  }

  /**
   * MapTokens turns vector of strings into numeric variables and puts them
   * into a given matrix. It is used as a helper function when trying to load
   * files. Each dimension's tokens are given in to this function. If one of the
   * tokens turns out to be a string, all the tokens should be mapped using the
   * MapString() funciton.
   *
   * @tparam eT Type of armadillo matrix.
   * @tparam MapType Type of unordered_map that contains mapped value pairs.
   * @param tokens Vector of variables inside a dimension.
   * @param row Position of the given tokens.
   * @param matrix Matrix to save the data into.
   * @param maps Maps given by the DatasetMapper class.
   * @param types Types of each dimensions given by the DatasetMapper class.
   */
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
