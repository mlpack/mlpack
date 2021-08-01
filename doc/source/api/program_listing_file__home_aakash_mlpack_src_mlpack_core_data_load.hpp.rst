
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_data_load.hpp:

Program Listing for File load.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_data_load.hpp>` (``/home/aakash/mlpack/src/mlpack/core/data/load.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_CORE_DATA_LOAD_HPP
   #define MLPACK_CORE_DATA_LOAD_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/log.hpp>
   #include <string>
   
   #include "format.hpp"
   #include "dataset_mapper.hpp"
   #include "image_info.hpp"
   
   namespace mlpack {
   namespace data  {
   
   template<typename eT>
   bool Load(const std::string& filename,
             arma::Mat<eT>& matrix,
             const bool fatal = false,
             const bool transpose = true,
             const arma::file_type inputLoadType = arma::auto_detect);
   
   template<typename eT>
   bool Load(const std::string& filename,
             arma::SpMat<eT>& matrix,
             const bool fatal = false,
             const bool transpose = true);
   
   extern template bool Load<int>(const std::string&,
                                  arma::Mat<int>&,
                                  const bool,
                                  const bool,
                                  const arma::file_type);
   
   // size_t and uword should be one of these three typedefs.
   extern template bool Load<unsigned int>(const std::string&,
                                           arma::Mat<unsigned int>&,
                                           const bool,
                                           const bool,
                                           const arma::file_type);
   
   extern template bool Load<unsigned long>(const std::string&,
                                            arma::Mat<unsigned long>&,
                                            const bool,
                                            const bool,
                                            const arma::file_type);
   
   extern template bool Load<unsigned long long>(const std::string&,
                                                 arma::Mat<unsigned long long>&,
                                                 const bool,
                                                 const bool,
                                                 const arma::file_type);
   
   extern template bool Load<float>(const std::string&,
                                    arma::Mat<float>&,
                                    const bool,
                                    const bool,
                                    const arma::file_type);
   
   extern template bool Load<double>(const std::string&,
                                     arma::Mat<double>&,
                                     const bool,
                                     const bool,
                                     const arma::file_type);
   
   extern template bool Load<int>(const std::string&,
                                  arma::Mat<int>&,
                                  const bool,
                                  const bool,
                                  const arma::file_type);
   
   extern template bool Load<unsigned int>(const std::string&,
                                           arma::SpMat<unsigned int>&,
                                           const bool,
                                           const bool);
   
   extern template bool Load<unsigned long>(const std::string&,
                                            arma::SpMat<unsigned long>&,
                                            const bool,
                                            const bool);
   
   extern template bool Load<unsigned long long>(const std::string&,
                                                 arma::SpMat<unsigned long long>&,
                                                 const bool,
                                                 const bool);
   
   extern template bool Load<float>(const std::string&,
                                    arma::SpMat<float>&,
                                    const bool,
                                    const bool);
   
   extern template bool Load<double>(const std::string&,
                                     arma::SpMat<double>&,
                                     const bool,
                                     const bool);
   
   template<typename eT>
   bool Load(const std::string& filename,
             arma::Col<eT>& vec,
             const bool fatal = false);
   
   template<typename eT>
   bool Load(const std::string& filename,
             arma::Row<eT>& rowvec,
             const bool fatal = false);
   
   template<typename eT, typename PolicyType>
   bool Load(const std::string& filename,
             arma::Mat<eT>& matrix,
             DatasetMapper<PolicyType>& info,
             const bool fatal = false,
             const bool transpose = true);
   
   extern template bool Load<int, IncrementPolicy>(
       const std::string&,
       arma::Mat<int>&,
       DatasetMapper<IncrementPolicy>&,
       const bool,
       const bool);
   
   extern template bool Load<arma::u32, IncrementPolicy>(
       const std::string&,
       arma::Mat<arma::u32>&,
       DatasetMapper<IncrementPolicy>&,
       const bool,
       const bool);
   
   extern template bool Load<arma::u64, IncrementPolicy>(
       const std::string&,
       arma::Mat<arma::u64>&,
       DatasetMapper<IncrementPolicy>&,
       const bool,
       const bool);
   
   extern template bool Load<float, IncrementPolicy>(
       const std::string&,
       arma::Mat<float>&,
       DatasetMapper<IncrementPolicy>&,
       const bool,
       const bool);
   
   extern template bool Load<double, IncrementPolicy>(
       const std::string&,
       arma::Mat<double>&,
       DatasetMapper<IncrementPolicy>&,
       const bool,
       const bool);
   
   template<typename T>
   bool Load(const std::string& filename,
             const std::string& name,
             T& t,
             const bool fatal = false,
             format f = format::autodetect);
   
   template<typename eT>
   bool Load(const std::string& filename,
             arma::Mat<eT>& matrix,
             ImageInfo& info,
             const bool fatal = false);
   
   template<typename eT>
   bool Load(const std::vector<std::string>& files,
             arma::Mat<eT>& matrix,
             ImageInfo& info,
             const bool fatal = false);
   
   // Implementation found in load_image.cpp.
   bool LoadImage(const std::string& filename,
                  arma::Mat<unsigned char>& matrix,
                  ImageInfo& info,
                  const bool fatal = false);
   
   } // namespace data
   } // namespace mlpack
   
   // Include implementation of model-loading Load() overload.
   #include "load_model_impl.hpp"
   // Include implementation of Load() for vectors.
   #include "load_vec_impl.hpp"
   // Include implementation of Load() for images.
   #include "load_image_impl.hpp"
   
   #endif
