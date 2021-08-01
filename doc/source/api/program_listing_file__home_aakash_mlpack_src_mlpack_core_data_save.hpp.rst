
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_save.hpp:

Program Listing for File save.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_save.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/save.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_SAVE_HPP
   #define MLPACK_CORE_DATA_SAVE_HPP
   
   #include <mlpack/core/util/log.hpp>
   #include <mlpack/core/arma_extend/arma_extend.hpp> // Includes Armadillo.
   #include <string>
   
   #include "format.hpp"
   #include "image_info.hpp"
   
   namespace mlpack {
   namespace data  {
   
   template<typename eT>
   bool Save(const std::string& filename,
             const arma::Mat<eT>& matrix,
             const bool fatal = false,
             bool transpose = true,
             arma::file_type inputSaveType = arma::auto_detect);
   
   template<typename eT>
   bool Save(const std::string& filename,
             const arma::SpMat<eT>& matrix,
             const bool fatal = false,
             bool transpose = true);
   
   template<typename T>
   bool Save(const std::string& filename,
             const std::string& name,
             T& t,
             const bool fatal = false,
             format f = format::autodetect);
   
   template<typename eT>
   bool Save(const std::string& filename,
             arma::Mat<eT>& matrix,
             ImageInfo& info,
             const bool fatal = false);
   
   template<typename eT>
   bool Save(const std::vector<std::string>& files,
             arma::Mat<eT>& matrix,
             ImageInfo& info,
             const bool fatal = false);
   
   bool SaveImage(const std::string& filename,
                  arma::Mat<unsigned char>& image,
                  ImageInfo& info,
                  const bool fatal = false);
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation.
   #include "save_impl.hpp"
   
   #endif
