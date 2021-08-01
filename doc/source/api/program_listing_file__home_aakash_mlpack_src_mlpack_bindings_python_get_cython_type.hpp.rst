
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_python_get_cython_type.hpp:

Program Listing for File get_cython_type.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_python_get_cython_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/python/get_cython_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_PYTHON_GET_CYTHON_TYPE_HPP
   #define MLPACK_BINDINGS_PYTHON_GET_CYTHON_TYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/core/util/is_std_vector.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace python {
   
   template<typename T>
   inline std::string GetCythonType(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0)
   {
     return "unknown";
   }
   
   template<>
   inline std::string GetCythonType<int>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<int>>::type*,
       const typename boost::disable_if<data::HasSerialize<int>>::type*,
       const typename boost::disable_if<arma::is_arma_type<int>>::type*)
   {
     return "int";
   }
   
   template<>
   inline std::string GetCythonType<double>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<double>>::type*,
       const typename boost::disable_if<data::HasSerialize<double>>::type*,
       const typename boost::disable_if<arma::is_arma_type<double>>::type*)
   {
     return "double";
   }
   
   template<>
   inline std::string GetCythonType<std::string>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<std::string>>::type*,
       const typename boost::disable_if<data::HasSerialize<std::string>>::type*,
       const typename boost::disable_if<arma::is_arma_type<std::string>>::type*)
   {
     return "string";
   }
   
   template<>
   inline std::string GetCythonType<size_t>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<size_t>>::type*,
       const typename boost::disable_if<data::HasSerialize<size_t>>::type*,
       const typename boost::disable_if<arma::is_arma_type<size_t>>::type*)
   {
     return "size_t";
   }
   
   template<>
   inline std::string GetCythonType<bool>(
       util::ParamData& /* d */,
       const typename boost::disable_if<util::IsStdVector<bool>>::type*,
       const typename boost::disable_if<data::HasSerialize<bool>>::type*,
       const typename boost::disable_if<arma::is_arma_type<bool>>::type*)
   {
     return "cbool";
   }
   
   template<typename T>
   inline std::string GetCythonType(
       util::ParamData& d,
       const typename boost::enable_if<util::IsStdVector<T>>::type* = 0)
   {
     return "vector[" + GetCythonType<typename T::value_type>(d) + "]";
   }
   
   template<typename T>
   inline std::string GetCythonType(
       util::ParamData& d,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     std::string type = "Mat";
     if (T::is_row)
       type = "Row";
     else if (T::is_col)
       type = "Col";
   
     return "arma." + type + "[" + GetCythonType<typename T::elem_type>(d) + "]";
   }
   
   template<typename T>
   inline std::string GetCythonType(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     return d.cppType + "*";
   }
   
   } // namespace python
   } // namespace bindings
   } // namespace mlpack
   
   #endif
