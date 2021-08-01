
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_import_decl.hpp:

Program Listing for File import_decl.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_import_decl.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/import_decl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_IMPORT_DECL_HPP
   #define MLPACK_BINDINGS_PYTHON_IMPORT_DECL_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "strip_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   void ImportDecl(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // First, we have to parse the type.  If we have something like, e.g.,
     // 'LogisticRegression<>', we must convert this to 'LogisticRegression[T=*].'
     std::string strippedType, printedType, defaultsType;
     StripType(d.cppType, strippedType, printedType, defaultsType);
   
     const std::string prefix = std::string(indent, ' ');
     std::cout << prefix << "cdef cppclass " << defaultsType << ":" << std::endl;
     std::cout << prefix << "  " << strippedType << "() nogil" << std::endl;
     std::cout << prefix << std::endl;
   }
   
   template<typename T>
   void ImportDecl(
       util::ParamData& /* d */,
       const size_t /* indent */,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Print nothing.
   }
   
   template<typename T>
   void ImportDecl(
       util::ParamData& /* d */,
       const size_t /* indent */,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // Print nothing.
   }
   
   template<typename T>
   void ImportDecl(util::ParamData& d,
                   const void* indent,
                   void* /* output */)
   {
     ImportDecl<typename std::remove_pointer<T>::type>(d, *((size_t*) indent));
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
