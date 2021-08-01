
.. _program_listing_file__home_aakash_mlpack_src_mlpack_core_util_size_checks.hpp:

Program Listing for File size_checks.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_core_util_size_checks.hpp>` (``/home/aakash/mlpack/src/mlpack/core/util/size_checks.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_UTIL_SIZE_CHECKS_HPP
   #define MLPACK_UTIL_SIZE_CHECKS_HPP
   
   
   namespace mlpack {
   namespace util {
   
   template<typename DataType, typename LabelsType>
   inline void CheckSameSizes(const DataType& data,
                              const LabelsType& label,
                              const std::string& callerDescription,
                              const std::string& addInfo = "labels")
   {
     if (data.n_cols != label.n_elem)
     {
       std::ostringstream oss;
       oss << callerDescription << ": number of points (" << data.n_cols << ") "
           << "does not match number of " << addInfo << " (" << label.n_elem
           << ")!" << std::endl;
       throw std::invalid_argument(oss.str());
     }
   }
   
   template<typename DataType>
   inline void CheckSameSizes(const DataType& data,
                              const size_t& size,
                              const std::string& callerDescription,
                              const std::string& addInfo = "labels")
   {
     if (data.n_cols != size)
     {
       std::ostringstream oss;
       oss << callerDescription << ": number of points (" << data.n_cols << ") "
           << "does not match number of " << addInfo << " (" << size << ")!"
           << std::endl;
       throw std::invalid_argument(oss.str());
     }
   }
   
   
   template<typename DataType, typename DimType>
   inline void CheckSameDimensionality(const DataType& data,
                                       const DimType& dimension,
                                       const std::string& callerDescription,
                                       const std::string& addInfo = "dataset")
   {
     if (data.n_rows != dimension.n_rows)
     {
       std::ostringstream oss;
       oss << callerDescription << ": dimensionality of " << addInfo << " ("
           << data.n_rows << ") is not equal to the dimensionality of the model"
           " (" << dimension.n_rows << ")!";
   
       throw std::invalid_argument(oss.str());
     }
   }
   
   template<typename DataType>
   inline void CheckSameDimensionality(const DataType& data,
                                       const size_t& dimension,
                                       const std::string& callerDescription,
                                       const std::string& addInfo = "dataset")
   {
     if (data.n_rows != dimension)
     {
       std::ostringstream oss;
       oss << callerDescription << ": dimensionality of " << addInfo << " ("
           << data.n_rows << ") is not equal to the dimensionality of the model"
           " (" << dimension << ")!";
       throw std::invalid_argument(oss.str());
     }
   }
   
   } // namespace util
   } // namespace mlpack
   
   #endif
