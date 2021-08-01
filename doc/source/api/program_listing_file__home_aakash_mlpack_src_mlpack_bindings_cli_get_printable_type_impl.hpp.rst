
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_type_impl.hpp:

Program Listing for File get_printable_type_impl.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_type_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/get_printable_type_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_IMPL_HPP
   #define MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_IMPL_HPP
   
   #include "get_printable_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::disable_if<util::IsStdVector<T>>::type*,
       const typename boost::disable_if<data::HasSerialize<T>>::type*,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type*)
   {
     if (std::is_same<T, bool>::value)
       return "flag";
     else if (std::is_same<T, int>::value)
       return "int";
     else if (std::is_same<T, double>::value)
       return "double";
     else if (std::is_same<T, std::string>::value)
       return "string";
     else
       throw std::invalid_argument("unknown parameter type" + data.cppType);
   }
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename std::enable_if<util::IsStdVector<T>::value>::type*)
   {
     if (std::is_same<T, std::vector<int>>::value)
       return "int vector";
     else if (std::is_same<T, std::vector<std::string>>::value)
       return "string vector";
     else
       throw std::invalid_argument("unknown vector type " + data.cppType);
   }
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
   {
     if (std::is_same<T, arma::mat>::value)
       return "2-d matrix file";
     else if (std::is_same<T, arma::Mat<size_t>>::value)
       return "2-d index matrix file";
     else if (std::is_same<T, arma::rowvec>::value)
       return "1-d matrix file";
     else if (std::is_same<T, arma::Row<size_t>>::value)
       return "1-d index matrix file";
     else if (std::is_same<T, arma::vec>::value)
       return "1-d matrix file";
     else if (std::is_same<T, arma::Col<size_t>>::value)
       return "1-d index matrix file";
     else
       throw std::invalid_argument("unknown Armadillo type" + data.cppType);
   }
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& /* data */,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
   {
     return "2-d categorical matrix file";
   }
   
   template<typename T>
   std::string GetPrintableType(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type*,
       const typename boost::enable_if<data::HasSerialize<T>>::type*)
   {
     return data.cppType + " file";
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
