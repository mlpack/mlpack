
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_detect_file_type.hpp:

Program Listing for File detect_file_type.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_detect_file_type.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/detect_file_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_DETECT_FILE_TYPE_HPP
   #define MLPACK_CORE_DATA_DETECT_FILE_TYPE_HPP
   
   namespace mlpack {
   namespace data {
   
   std::string GetStringType(const arma::file_type& type);
   
   arma::file_type GuessFileType(std::istream& f);
   
   arma::file_type AutoDetect(std::fstream& stream,
                              const std::string& filename);
   
   arma::file_type DetectFromExtension(const std::string& filename);
   
   } // namespace data
   } // namespace mlpack
   
   #endif
