
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_get_printable_param.hpp:

Program Listing for File get_printable_param.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_get_printable_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/get_printable_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_GET_PRINTABLE_PARAM_HPP
   #define MLPACK_BINDINGS_R_GET_PRINTABLE_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     std::ostringstream oss;
     oss << boost::any_cast<T>(data.value);
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0)
   {
     const T& t = boost::any_cast<T>(data.value);
   
     std::ostringstream oss;
     for (size_t i = 0; i < t.size(); ++i)
       oss << t[i] << " ";
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // Get the matrix.
     const T& matrix = boost::any_cast<T>(data.value);
   
     std::ostringstream oss;
     oss << matrix.n_rows << "x" << matrix.n_cols << " matrix";
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     std::ostringstream oss;
     oss << data.cppType << " model at " << boost::any_cast<T*>(data.value);
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     // Get the matrix.
     const T& tuple = boost::any_cast<T>(data.value);
     const arma::mat& matrix = std::get<1>(tuple);
   
     std::ostringstream oss;
     oss << matrix.n_rows << "x" << matrix.n_cols << " matrix with dimension type "
         << "information";
     return oss.str();
   }
   
   template<typename T>
   void GetPrintableParam(util::ParamData& data,
                          const void* /* input */,
                          void* output)
   {
     *((std::string*) output) =
         GetPrintableParam<typename std::remove_pointer<T>::type>(data);
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
