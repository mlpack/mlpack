/**
 * @file core/data/map_policies/missing_policy.hpp
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

#include <mlpack/prereqs.hpp>
#include <unordered_map>
#include <mlpack/core/data/map_policies/datatype.hpp>
#include <limits>
#include <set>

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

  //! This doesn't need a first pass over the data to set up.
  static const bool NeedsFirstPass = false;

  /**
   * There is nothing for us to do here, but this is required by the MapPolicy
   * type.
   */
  template<typename T>
  void MapFirstPass(const std::string& /* string */, const size_t /* dim */)
  {
    // Nothing to do.
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
   * @param * (types) Vector containing the type information about each
   *          dimensions.
   */
  template<typename MapType, typename T>
  T MapString(const std::string& string,
              const size_t dimension,
              MapType& maps,
              std::vector<Datatype>& /* types */)
  {
    static_assert(std::numeric_limits<T>::has_quiet_NaN == true,
        "Cannot use MissingPolicy with types where has_quiet_NaN() is false!");

    // If we can load the string then there is no need for mapping.
    std::stringstream token;
    token.str(string);
    T t;
    token >> t; // Could be sped up by only doing this if we need to.

    MappedType value = std::numeric_limits<MappedType>::quiet_NaN();
    // But we can't use that for the map, so we need some other thing that will
    // represent quiet_NaN().
    const MappedType mapValue = std::nexttoward(
        std::numeric_limits<MappedType>::max(), MappedType(0));

    // If extraction of the value fails, or if it is a value that is supposed to
    // be mapped, then do mapping.
    if (token.fail() || !token.eof() ||
        missingSet.find(string) != std::end(missingSet))
    {
      // Everything is mapped to NaN.  However we must still keep track of
      // everything that we have mapped, so we add it to the maps if needed.
      if (maps.count(dimension) == 0 ||
          maps[dimension].first.count(string) == 0)
      {
        // This string does not exist yet.
        using PairType = std::pair<std::string, MappedType>;
        maps[dimension].first.insert(PairType(string, value));

        // Insert right mapping too.
        if (maps[dimension].second.count(mapValue) == 0)
        {
          // Create new element in reverse map.
          maps[dimension].second.insert(std::make_pair(mapValue,
              std::vector<std::string>()));
        }
        maps[dimension].second[mapValue].push_back(string);
      }

      return value;
    }
    else
    {
      // We can just return the value that we read.
      return t;
    }
  }

 private:
  // Note that missingSet and maps are different.
  // missingSet specifies which value/string should be mapped and may be a
  // superset of 'maps'.
  std::set<std::string> missingSet;
}; // class MissingPolicy

} // namespace data
} // namespace mlpack

#endif
