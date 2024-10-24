/**
 * @file core/data/dataset_mapper.hpp
 * @author Ryan Curtin
 * @author Keon Kim
 *
 * Defines the DatasetMapper class, which holds information about a dataset.
 * This is useful when the dataset contains categorical non-numeric features
 * that needs to be mapped to categorical numeric features.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_DATA_DATASET_INFO_HPP
#define MLPACK_CORE_DATA_DATASET_INFO_HPP

#include <mlpack/prereqs.hpp>
#include <unordered_map>

#include "map_policies/increment_policy.hpp"

namespace mlpack {
namespace data {

/**
 * Auxiliary information for a dataset, including mappings to/from strings (or
 * other types) and the datatype of each dimension.  DatasetMapper objects are
 * optionally produced by data::Load(), and store the type of each dimension
 * (Datatype::numeric or Datatype::categorical) as well as mappings from strings
 * to unsigned integers and vice versa.
 *
 * DatasetMapper objects can also map from arbitrary types; the type to map from
 * can be specified with the InputType template parameter.  By default, the
 * InputType parameter is std::string.
 *
 * @tparam PolicyType Mapping policy used to specify MapString().
 * @tparam InputType Type of input to be mapped.
 */
template<typename PolicyType, typename InputType = std::string>
class DatasetMapper
{
 public:
  /**
   * Create the DatasetMapper object with the given dimensionality.  Note that
   * the dimensionality cannot be changed later; you will have to create a new
   * DatasetMapper object.
   */
  explicit DatasetMapper(const size_t dimensionality = 0);

  /**
   * Create the DatasetMapper object with the given policy and dimensionality.
   * Note that the dimensionality cannot be changed later; you will have to
   * create a new DatasetMapper object. Policy can be modified by the modifier.
   */
  explicit DatasetMapper(PolicyType& policy, const size_t dimensionality = 0);

  /**
   * Set the dimensionality of an existing DatasetMapper object.  This resets
   * all mappings (but not the PolicyType).
   *
   * @param dimensionality New dimensionality.
   */
  void SetDimensionality(const size_t dimensionality);

  /**
   * Preprocessing: during a first pass of the data, pass the input on to the
   * MapPolicy if they are needed.
   *
   * @param input Input to map.
   * @param dimension Dimension to map for.
   */
  template<typename T>
  void MapFirstPass(const InputType& input, const size_t dimension);

  /**
   * Given the input and the dimension to which it belongs, return its numeric
   * mapping.  If no mapping yet exists, the input is added to the list of
   * mappings for the given dimension.  The dimension parameter refers to the
   * index of the dimension of the string (i.e. the row in the dataset).
   *
   * @tparam T Numeric type to map to (int/double/float/etc.).
   * @param input Input to find/create mapping for.
   * @param dimension Index of the dimension of the string.
   */
  template<typename T>
  T MapString(const InputType& input,
              const size_t dimension);

  /**
   * Return the input that corresponds to a given value in a given dimension.
   * If the value is not a valid mapping in the given dimension, a
   * std::invalid_argument is thrown.  Note that this does not remove the
   * mapping.
   *
   * If the mapping is non-unique (i.e. many strings can map to the same value),
   * then you can pass a different value for unmappingIndex to get a different
   * string that maps to the given value.  unmappingIndex should be in the range
   * from 0 to (NumUnmappings(value, dimension) - 1).
   *
   * If the mapping is unique (which it is for DatasetInfo), then the
   * unmappingIndex parameter can be left as the default.
   *
   * @param value Mapped value for input.
   * @param dimension Dimension to unmap string from.
   * @param unmappingIndex Index of non-unique unmapping (optional).
   */
  template<typename T>
  const InputType& UnmapString(const T value,
                               const size_t dimension,
                               const size_t unmappingIndex = 0) const;

  /**
   * Get the number of possible unmappings for a string in a given dimension.
   */
  template<typename T>
  size_t NumUnmappings(const T value, const size_t dimension) const;

  /**
   * Return the value that corresponds to a given input in a given dimension.
   * If the value is not a valid mapping in the given dimension, a
   * std::invalid_argument is thrown.  Note that this does not remove the
   * mapping.
   *
   * @param input Mapped input for value.
   * @param dimension Dimension to unmap input from.
   */
  typename PolicyType::MappedType UnmapValue(const InputType& input,
                                             const size_t dimension);

  //! Return the type of a given dimension (numeric or categorical).
  Datatype Type(const size_t dimension) const;
  //! Modify the type of a given dimension (be careful!).
  Datatype& Type(const size_t dimension);

  /**
   * Get the number of mappings for a particular dimension.  If the dimension
   * is numeric, then this will return 0.
   */
  size_t NumMappings(const size_t dimension) const;

  /**
   * Get the dimensionality of the DatasetMapper object (that is, how many
   * dimensions it has information for).  If this object was created by a call
   * to mlpack::data::Load(), then the dimensionality will be the same as the
   * number of rows (dimensions) in the dataset.
   */
  size_t Dimensionality() const;

  /**
   * Serialize the dataset information.
   */
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(types));
    ar(CEREAL_NVP(maps));
  }

  //! Return the policy of the mapper.
  const PolicyType& Policy() const;

  //! Modify the policy of the mapper (be careful!).
  PolicyType& Policy();
  //! Modify (Replace) the policy of the mapper with a new policy.
  void Policy(PolicyType&& policy);

 private:
  //! Types of each dimension.
  std::vector<Datatype> types;

  // Forward mapping type.
  using ForwardMapType = std::unordered_map<InputType,
      typename PolicyType::MappedType>;

  // Reverse mapping type.  Multiple inputs may map to a single output, hence
  // the need for std::vector.
  using ReverseMapType = std::unordered_map<typename PolicyType::MappedType,
      std::vector<InputType>>;

  // Mappings from strings to integers.
  // Map entries will only exist for dimensions that are categorical.
  // MapType = map<dimension, pair<bimap<string, MappedType>, numMappings>>
  using MapType = std::unordered_map<size_t, std::pair<ForwardMapType,
      ReverseMapType>>;

  //! maps object stores string and numerical pairs.
  MapType maps;

  //! policy object tells dataset mapper how the categorical values should be
  //  mapped to the maps object. It is used in MapString() and MapTokens().
  PolicyType policy;
};

// Use typedef to provide backward compatibility
using DatasetInfo = DatasetMapper<data::IncrementPolicy>;

} // namespace data
} // namespace mlpack

#include "dataset_mapper_impl.hpp"

// Also include utility function.
#include "check_categorical_param.hpp"

#endif
