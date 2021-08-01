
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_map_policies_datatype.hpp:

Program Listing for File datatype.hpp
=====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_map_policies_datatype.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/map_policies/datatype.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_MAP_POLICIES_DATATYPE_HPP
   #define MLPACK_CORE_DATA_MAP_POLICIES_DATATYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   enum Datatype : bool /* [> bool is all the precision we need for two types <] */
   {
     numeric = 0,
     categorical = 1
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif
