/**
 * @file missing_policy.hpp
 * @author Keon Kim
 *
 * Missing map policy for dataset info.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_MAP_POLICIES_MISSING_POLICY_HPP
#define MLPACK_CORE_DATA_MAP_POLICIES_MISSING_POLICY_HPP

#include <mlpack/core.hpp>
#include <unordered_map>
#include <boost/bimap.hpp>
#include <mlpack/core/data/map_policies/datatype.hpp>
#include <limits>

namespace mlpack {
namespace data {
/**
 * MissingPolicy is used as a helper class for DatasetMapper. It tells how the
 * strings should be mapped. Purpose of this policy is to map all user-defined
 * missing variables into maps so that users can decide what to do with the
 * corrupted data. User-defined missing variables are given by the missingSet.
 * Note that MissingPolicy does not change type of features.
 */
class MissingPolicy
{
 public:
  // typedef of MappedType
  using MappedType = double;

  MissingPolicy()
  {
    // Nothing to initialize here.
  }

  /**
   * Create the MissingPolicy object with the given missingSet. Note that the
   * missingSet cannot be changed later; you will have to create a new
   * MissingPolicy object.
   *
   * @param missingSet Set of strings that should be mapped.
   */
  explicit MissingPolicy(std::set<std::string> missingSet) :
      missingSet(std::move(missingSet))
  {
    // Nothing to initialize here.
  }

  /**
   * Given the string and the dimension to which it belongs by the user, and
   * the maps and types given by the DatasetMapper class, returns its numeric
   * mapping. If no mapping yet exists and the string is included in the
   * missingSet, the string is added to the list of mappings for the given
   * dimension. This function is used as a helper function for DatasetMapper
   * class.
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
    // mute the unused parameter warning (does nothing here.)
    (void)types;
    // If this condition is true, either we have no mapping for the given string
    // or we have no mappings for the given dimension at all.  In either case,
    // we create a mapping.
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    if (missingSet.count(string) != 0 &&
        (maps.count(dimension) == 0 ||
         maps[dimension].first.left.count(string) == 0))
    {
      // This string does not exist yet.
      typedef boost::bimap<std::string, MappedType>::value_type PairType;
      maps[dimension].first.insert(PairType(string, NaN));

      size_t& numMappings = maps[dimension].second;
      ++numMappings;
      return NaN;
    }
    else
    {
      // This string already exists in the mapping or not included in
      // the missingSet.
      return NaN;
    }
  }

  /**
   * MapTokens turns vector of strings into numeric variables and puts them
   * into a given matrix. It is used as a helper function when trying to load
   * files. Each dimension's tokens are given in to this function. If one of the
   * tokens turns out to be a string or one of the missingSet's variables, only
   * the token responsible for it should be mapped using the MapString()
   * funciton.
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
    // MissingPolicy allows double type matrix only, because it uses NaN.
    static_assert(std::is_same<eT, double>::value, "You must use double type "
        " matrix in order to apply MissingPolicy");

    std::stringstream token;
    for (size_t i = 0; i != tokens.size(); ++i)
    {
      token.str(tokens[i]);
      token>>matrix.at(row, i);
      // if the token is not number, map it.
      // or if token is a number, but is included in the missingSet, map it.
      if (token.fail() || missingSet.find(tokens[i]) != std::end(missingSet))
      {
        const eT val = static_cast<eT>(this->MapString(tokens[i], row, maps,
                                                       types));
        matrix.at(row, i) = val;
      }
      token.clear();
    }
  }

 private:
  // Note that missingSet and maps are different.
  // missingSet specifies which value/string should be mapped.
  std::set<std::string> missingSet;
}; // class MissingPolicy

} // namespace data
} // namespace mlpack

#endif
