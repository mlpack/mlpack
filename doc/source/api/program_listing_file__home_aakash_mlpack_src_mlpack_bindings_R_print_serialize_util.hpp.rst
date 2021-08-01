
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_R_print_serialize_util.hpp:

Program Listing for File print_serialize_util.hpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_R_print_serialize_util.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/R/print_serialize_util.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_R_PRINT_SERIALIZE_UTIL_HPP
   #define MLPACK_BINDINGS_R_PRINT_SERIALIZE_UTIL_HPP
   
   #include <mlpack/bindings/util/strip_type.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace r {
   
   template<typename T>
   void PrintSerializeUtil(
       util::ParamData& /* d */,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0)
   {
     // Do Nothing.
   }
   
   template<typename T>
   void PrintSerializeUtil(
       util::ParamData& /* d */,
       const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
   {
     // Do Nothing.
   }
   
   template<typename T>
   void PrintSerializeUtil(
       util::ParamData& d,
       const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
       const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
   {
     MLPACK_COUT_STREAM << "  " << d.name << " <- IO_GetParam"
         << util::StripType(d.cppType) << "Ptr(\"" << d.name << "\")";
     MLPACK_COUT_STREAM << std::endl;
     MLPACK_COUT_STREAM << "  attr(" << d.name << ", \"type\") <- \""
         << util::StripType(d.cppType) << "\"";
     MLPACK_COUT_STREAM << std::endl;
   }
   
   template<typename T>
   void PrintSerializeUtil(util::ParamData& d,
                           const void* /*input*/,
                           void* /* output */)
   {
     PrintSerializeUtil<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace r
   } // namespace bindings
   } // namespace mlpack
   
   #endif
