
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_param_impl.hpp:

Program Listing for File get_printable_param_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_get_printable_param_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/get_printable_param_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_IMPL_HPP
   #define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_IMPL_HPP
   
   #include "get_printable_param.hpp"
   #include "get_param.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
       const typename boost::disable_if<util::IsStdVector<T>>::type* /* junk */,
       const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
   {
     std::ostringstream oss;
     oss << boost::any_cast<T>(data.value);
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename std::enable_if<util::IsStdVector<T>::value>::type*
           /* junk */)
   {
     const T& t = boost::any_cast<T>(data.value);
   
     std::ostringstream oss;
     for (size_t i = 0; i < t.size(); ++i)
       oss << t[i] << " ";
     return oss.str();
   }
   
   // Return a printed representation of the size of the matrix.
   template<typename T>
   std::string GetMatrixSize(
       T& matrix,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
   {
     std::ostringstream oss;
     oss << matrix.n_rows << "x" << matrix.n_cols << " matrix";
     return oss.str();
   }
   
   // Return a printed representation of the size of the matrix.
   template<typename T>
   std::string GetMatrixSize(
       T& matrixAndInfo,
       const typename std::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
   {
     return GetMatrixSize(std::get<1>(matrixAndInfo));
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename std::enable_if<arma::is_arma_type<T>::value ||
                                     std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* /* junk */)
   {
     // Extract the string from the tuple that's being held.
     typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
     const TupleType* tuple = boost::any_cast<TupleType>(&data.value);
   
     std::ostringstream oss;
     oss << "'" << std::get<0>(std::get<1>(*tuple)) << "'";
   
     if (std::get<0>(std::get<1>(*tuple)) != "")
     {
       // Make sure the matrix is loaded so that we can print its size.
       GetParam<T>(const_cast<util::ParamData&>(data));
       std::string matDescription =
           std::to_string(std::get<2>(std::get<1>(*tuple))) + "x" +
           std::to_string(std::get<1>(std::get<1>(*tuple))) + " matrix";
   
       oss << " (" << matDescription << ")";
     }
   
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
       const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
   {
     // Extract the string from the tuple that's being held.
     typedef std::tuple<T*, typename ParameterType<T>::type> TupleType;
     const TupleType* tuple = boost::any_cast<TupleType>(&data.value);
   
     std::ostringstream oss;
     oss << std::get<1>(*tuple);
     return oss.str();
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
