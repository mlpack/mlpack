
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_default_param_impl.hpp:

Program Listing for File default_param_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_default_param_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/default_param_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_DEFAULT_PARAM_IMPL_HPP
   #define MLPACK_BINDINGS_R_DEFAULT_PARAM_IMPL_HPP
   
   #include "default_param.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
       const typename boost::disable_if<util::IsStdVector<T>>::type* /* junk */,
       const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
       const typename boost::disable_if<std::is_same<T, std::string>>::type*,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* /* junk */)
   {
     std::ostringstream oss;
     if (std::is_same<T, bool>::value)
       oss << "FALSE";
     else
       oss << boost::any_cast<T>(data.value);
   
     return oss.str();
   }
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<util::IsStdVector<T>>::type* /* junk */)
   {
     // Print each element in an array delimited by square brackets.
     std::ostringstream oss;
     const T& vector = boost::any_cast<T>(data.value);
     oss << "c(";
     if (std::is_same<T, std::vector<std::string>>::value)
     {
       if (vector.size() > 0)
       {
         for (size_t i = 0; i < vector.size() - 1; ++i)
         {
           oss << "'" << vector[i] << "', ";
         }
   
         oss << "'" << vector[vector.size() - 1] << "'";
       }
   
       oss << ")";
     }
     else
     {
       if (vector.size() > 0)
       {
         for (size_t i = 0; i < vector.size() - 1; ++i)
         {
           oss << vector[i] << ", ";
         }
   
         oss << vector[vector.size() - 1];
       }
   
       oss << ")";
     }
     return oss.str();
   }
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& data,
       const typename boost::enable_if<std::is_same<T, std::string>>::type*)
   {
     const std::string& s = *boost::any_cast<std::string>(&data.value);
     return "\"" + s + "\"";
   }
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& /* data */,
       const typename boost::enable_if_c<
           arma::is_arma_type<T>::value ||
           std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                      arma::mat>>::value>::type* /* junk */)
   {
     // Get the filename and return it, or return an empty string.
     if (std::is_same<T, arma::rowvec>::value ||
         std::is_same<T, arma::vec>::value ||
         std::is_same<T, arma::mat>::value)
     {
       return "matrix(numeric(), 0, 0)";
     }
     else if (std::is_same<T, arma::Row<size_t>>::value ||
         std::is_same<T, arma::Col<size_t>>::value ||
         std::is_same<T, arma::Mat<size_t>>::value)
     {
       return "matrix(integer(), 0, 0)";
     }
     else
     {
       return "matrix(numeric(), 0, 0)";
     }
   }
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& /* data */,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
       const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
   {
     return "NA";
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
