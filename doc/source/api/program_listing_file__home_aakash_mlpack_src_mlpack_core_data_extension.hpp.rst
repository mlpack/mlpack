
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_extension.hpp:

Program Listing for File extension.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_extension.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/extension.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_EXTENSION_HPP
   #define MLPACK_CORE_DATA_EXTENSION_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace data {
   
   inline std::string Extension(const std::string& filename)
   {
     const size_t ext = filename.rfind('.');
     std::string extension;
     if (ext == std::string::npos)
       return extension;
   
     extension = filename.substr(ext + 1);
     std::transform(extension.begin(), extension.end(), extension.begin(),
         ::tolower);
   
     return extension;
   }
   
   } // namespace data
   } // namespace mlpack
   
   #endif
