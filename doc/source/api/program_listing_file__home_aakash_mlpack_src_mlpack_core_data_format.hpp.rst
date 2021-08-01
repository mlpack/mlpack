
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_format.hpp:

Program Listing for File format.hpp
===================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_format.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/format.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_FORMATS_HPP
   #define MLPACK_CORE_DATA_FORMATS_HPP
   
   namespace mlpack {
   namespace data {
   
   enum format
   {
     autodetect,
     json,
     xml,
     binary
   };
   
   } // namespace data
   } // namespace mlpack
   
   #endif
