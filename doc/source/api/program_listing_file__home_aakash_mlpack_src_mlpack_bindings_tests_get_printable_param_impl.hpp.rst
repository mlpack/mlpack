
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_tests_get_printable_param_impl.hpp:

Program Listing for File get_printable_param_impl.hpp
=====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_tests_get_printable_param_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/tests/get_printable_param_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_TESTS_GET_PRINTABLE_PARAM_IMPL_HPP
   #define MLPACK_BINDINGS_TESTS_GET_PRINTABLE_PARAM_IMPL_HPP
   
   #include "get_printable_param.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace tests {
   
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
       const typename boost::enable_if<util::IsStdVector<T>>::type* /* junk */)
   {
     const T& t = boost::any_cast<T>(data.value);
   
     std::ostringstream oss;
     for (size_t i = 0; i < t.size(); ++i)
       oss << t[i] << " ";
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& /* data */,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
   {
     return "matrix type";
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& data,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
       const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
   {
     // Extract the string from the tuple that's being held.
     std::ostringstream oss;
     oss << data.cppType << " model";
     return oss.str();
   }
   
   template<typename T>
   std::string GetPrintableParam(
       util::ParamData& /* data */,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
   {
     return "matrix/DatatsetInfo tuple";
   }
   
   } // namespace tests
   } // namespace bindings
   } // namespace mlpack
   
   #endif
