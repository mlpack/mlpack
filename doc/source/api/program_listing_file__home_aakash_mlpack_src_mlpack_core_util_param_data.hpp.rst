
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_param_data.hpp:

Program Listing for File param_data.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_param_data.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/param_data.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_UTIL_PARAM_DATA_HPP
   #define MLPACK_CORE_UTIL_PARAM_DATA_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <boost/any.hpp>
   
   #define TYPENAME(x) (std::string(typeid(x).name()))
   
   namespace mlpack {
   namespace data {
   
   class IncrementPolicy;
   
   template<typename PolicyType, typename InputType>
   class DatasetMapper;
   
   using DatasetInfo = DatasetMapper<IncrementPolicy, std::string>;
   
   } // namespace data
   } // namespace mlpack
   
   namespace mlpack {
   namespace util {
   
   struct ParamData
   {
     std::string name;
     std::string desc;
     std::string tname;
     char alias;
     bool wasPassed;
     bool noTranspose;
     bool required;
     bool input;
     bool loaded;
     bool persistent;
     boost::any value;
     std::string cppType;
   };
   
   } // namespace util
   } // namespace mlpack
   
   #endif
