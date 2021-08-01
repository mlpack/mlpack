
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_imputer.hpp:

Program Listing for File imputer.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_imputer.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/imputer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_IMPUTER_HPP
   #define MLPACK_CORE_DATA_IMPUTER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "dataset_mapper.hpp"
   #include "map_policies/missing_policy.hpp"
   #include "map_policies/increment_policy.hpp"
   
   namespace mlpack {
   namespace data {
   
   template<typename T, typename MapperType, typename StrategyType>
   class Imputer
   {
    public:
     Imputer(MapperType mapper, bool columnMajor = true):
         mapper(std::move(mapper)),
         columnMajor(columnMajor)
     {
       // Nothing to initialize here.
     }
   
     Imputer(MapperType mapper, StrategyType strategy, bool columnMajor = true):
         strategy(std::move(strategy)),
         mapper(std::move(mapper)),
         columnMajor(columnMajor)
     {
       // Nothing to initialize here.
     }
   
     void Impute(arma::Mat<T>& input,
                 const std::string& missingValue,
                 const size_t dimension)
     {
       T mappedValue = static_cast<T>(mapper.UnmapValue(missingValue, dimension));
       strategy.Impute(input, mappedValue, dimension, columnMajor);
     }
   
     const StrategyType& Strategy() const { return strategy; }
   
     StrategyType& Strategy() { return strategy; }
   
     const MapperType& Mapper() const { return mapper; }
   
     MapperType& Mapper() { return mapper; }
   
    private:
     // StrategyType
     StrategyType strategy;
   
     // DatasetMapperType<MapPolicy>
     MapperType mapper;
   
     // save columnMajor as a member variable since it is rarely changed.
     bool columnMajor;
   }; // class Imputer
   
   } // namespace data
   } // namespace mlpack
   
   #endif
