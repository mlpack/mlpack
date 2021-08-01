
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_dataset_mapper.hpp:

Program Listing for File dataset_mapper.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_dataset_mapper.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/dataset_mapper.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_DATASET_INFO_HPP
   #define MLPACK_CORE_DATA_DATASET_INFO_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <unordered_map>
   
   #include "map_policies/increment_policy.hpp"
   
   namespace mlpack {
   namespace data {
   
   template<typename PolicyType, typename InputType = std::string>
   class DatasetMapper
   {
    public:
     explicit DatasetMapper(const size_t dimensionality = 0);
   
     explicit DatasetMapper(PolicyType& policy, const size_t dimensionality = 0);
   
     void SetDimensionality(const size_t dimensionality);
   
     template<typename T>
     void MapFirstPass(const InputType& input, const size_t dimension);
   
     template<typename T>
     T MapString(const InputType& input,
                 const size_t dimension);
   
     template<typename T>
     const InputType& UnmapString(const T value,
                                  const size_t dimension,
                                  const size_t unmappingIndex = 0) const;
   
     template<typename T>
     size_t NumUnmappings(const T value, const size_t dimension) const;
   
     typename PolicyType::MappedType UnmapValue(const InputType& input,
                                                const size_t dimension);
   
     Datatype Type(const size_t dimension) const;
     Datatype& Type(const size_t dimension);
   
     size_t NumMappings(const size_t dimension) const;
   
     size_t Dimensionality() const;
   
     template<typename Archive>
     void serialize(Archive& ar, const uint32_t /* version */)
     {
       ar(CEREAL_NVP(types));
       ar(CEREAL_NVP(maps));
     }
   
     const PolicyType& Policy() const;
   
     PolicyType& Policy();
     void Policy(PolicyType&& policy);
   
    private:
     std::vector<Datatype> types;
   
     // Forward mapping type.
     using ForwardMapType = typename std::unordered_map<InputType, typename
         PolicyType::MappedType>;
   
     // Reverse mapping type.  Multiple inputs may map to a single output, hence
     // the need for std::vector.
     using ReverseMapType = std::unordered_map<typename PolicyType::MappedType,
         std::vector<InputType>>;
   
     // Mappings from strings to integers.
     // Map entries will only exist for dimensions that are categorical.
     // MapType = map<dimension, pair<bimap<string, MappedType>, numMappings>>
     using MapType = std::unordered_map<size_t, std::pair<ForwardMapType,
         ReverseMapType>>;
   
     MapType maps;
   
     //  mapped to the maps object. It is used in MapString() and MapTokens().
     PolicyType policy;
   };
   
   // Use typedef to provide backward compatibility
   using DatasetInfo = DatasetMapper<data::IncrementPolicy>;
   
   } // namespace data
   } // namespace mlpack
   
   #include "dataset_mapper_impl.hpp"
   
   #endif
