/**
 * @file core/data/map_policies/increment_policy.hpp
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

#include <mlpack/prereqs.hpp>
#include <unordered_map>
#include <mlpack/core/data/map_policies/datatype.hpp>

namespace mlpack {
namespace data {

/**
 * IncrementPolicy is used as a helper class for DatasetMapper. It tells how the
 * strings should be mapped. Purpose of this policy is to map all dimension if
 * one of the variables in a dimension turns out to be a categorical variable.
 * IncrementPolicy maps strings to incrementing unsigned integers (size_t).
 * The first input to be mapped will be mapped to 0, the next to 1 and so on.
 *
 * If the 'forceAllMappings' parameter is set to true, this will always map.
 * Otherwise, inputs will only be mapped if they cannot be cast to the output
 * type via a stringstream extraction.
 */
class IncrementPolicy
{
 public:
  IncrementPolicy(const bool forceAllMappings = false) :
      forceAllMappings(forceAllMappings) { }

  // typedef of MappedType
  using MappedType = size_t;

  //! We do need a first pass over the data to set the dimension types right.
  static const bool NeedsFirstPass = true;

  /**
   * Determine if the dimension is numeric or categorical.
   */
  template<typename T, typename InputType>
  void MapFirstPass(const InputType& input,
                    const size_t dim,
                    std::vector<Datatype>& types)
  {
    if (types[dim] == Datatype::categorical)
    {
      // No need to check; it's already categorical.
      return;
    }

    if (forceAllMappings)
    {
      types[dim] = Datatype::categorical;
    }
    else
    {
      // Attempt to convert the input to an output type via a stringstream.
      std::stringstream token;
      token << input;
      T val;
      token >> val;

      if (token.fail() || !token.eof())
        types[dim] = Datatype::categorical;
    }
  }

  /**
   * Given the input and the dimension to which the it belongs, and the maps
   * and types given by the DatasetMapper class, returns its numeric mapping.
   * If no mapping yet exists, the input is added to the list of mappings for
   * the given dimension. This function is used as a helper function for
   * DatasetMapper class.
   *
   * @tparam MapType Type of unordered_map that contains mapped value pairs
   * @param input Input to find/create mapping for.
   * @param dimension Index of the dimension of the input.
   * @param maps Unordered map given by the DatasetMapper.
   * @param types Vector containing the type information about each dimensions.
   */
  template<typename MapType, typename T, typename InputType>
  T MapString(const InputType& input,
              const size_t dimension,
              MapType& maps,
              std::vector<Datatype>& types)
  {
    // If we are in a categorical dimension we already know we need to map.
    if (types[dimension] == Datatype::numeric && !forceAllMappings)
    {
      // Check if this input needs to be mapped or if it can be read
      // directly as a number.  This will be true if nothing else in this
      // dimension has yet been mapped, but this can't be read as a number.
      std::stringstream token;
      token << input;
      T val;
      token >> val;

      if (!token.fail() && token.eof())
        return val;

      // Otherwise, we must map.
    }

    // If this condition is true, either we have no mapping for the given input
    // or we have no mappings for the given dimension at all.  In either case,
    // we create a mapping.
    if (maps.count(dimension) == 0 ||
        maps[dimension].first.count(input) == 0)
    {
      // This input does not exist yet.
      size_t numMappings = maps[dimension].first.size();

      // Change type of the feature to categorical.
      if (numMappings == 0)
        types[dimension] = Datatype::categorical;

      using PairType = std::pair<InputType, MappedType>;
      maps[dimension].first.insert(PairType(input, numMappings));

      // Do we need to create the second map?
      if (maps[dimension].second.count(numMappings) == 0)
      {
        maps[dimension].second.insert(std::make_pair(numMappings,
            std::vector<InputType>()));
      }
      maps[dimension].second[numMappings].push_back(input);

      return T(numMappings);
    }
    else
    {
      // This input already exists in the mapping.
      return maps[dimension].first.at(input);
    }
  }

 private:
  // Whether or not we should map all tokens.
  bool forceAllMappings;
}; // class IncrementPolicy

} // namespace data
} // namespace mlpack

#endif
