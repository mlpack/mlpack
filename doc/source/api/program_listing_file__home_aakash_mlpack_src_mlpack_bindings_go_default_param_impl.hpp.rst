
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_default_param_impl.hpp:

Program Listing for File default_param_impl.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_default_param_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/default_param_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_DEFAULT_PARAM_IMPL_HPP
   #define MLPACK_BINDINGS_GO_DEFAULT_PARAM_IMPL_HPP
   
   #include "default_param.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
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
       oss << "false";
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
     if (std::is_same<T, std::vector<std::string>>::value)
     {
       oss << "[]string{";
       if (vector.size() > 0)
       {
         for (size_t i = 0; i < vector.size() - 1; ++i)
         {
           oss << "\"" << vector[i] << "\", ";
         }
   
         oss << "\"" << vector[vector.size() - 1] << "\"";
       }
   
       oss << "}";
     }
     else if (std::is_same<T, std::vector<int>>::value)
     {
       oss << "[]int{";
       if (vector.size() > 0)
       {
         for (size_t i = 0; i < vector.size() - 1; ++i)
         {
           oss << vector[i] << ", ";
         }
   
         oss << vector[vector.size() - 1];
       }
   
       oss << "}";
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
         std::is_same<T, arma::vec>::value)
     {
       return "mat.NewDense(1, 1, nil)";
     }
     else if (std::is_same<T, arma::Col<size_t>>::value ||
              std::is_same<T, arma::Row<size_t>>::value)
     {
       return "mat.NewDense(1, 1, nil)";
     }
     else if (std::is_same<T, arma::Mat<size_t>>::value)
     {
       return "mat.NewDense(1, 1, nil)";
     }
     else
     {
       return "mat.NewDense(1, 1, nil)";
     }
   }
   
   template<typename T>
   std::string DefaultParamImpl(
       util::ParamData& /* data */,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
       const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
   {
     return "nil";
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
