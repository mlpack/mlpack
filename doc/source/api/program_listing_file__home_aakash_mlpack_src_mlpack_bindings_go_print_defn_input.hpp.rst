
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_defn_input.hpp:

Program Listing for File print_defn_input.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_defn_input.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_defn_input.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_DEFN_INPUT_HPP
   #define MLPACK_BINDINGS_GO_PRINT_DEFN_INPUT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/bindings/util/camel_case.hpp>
   #include "get_go_type.hpp"
   #include "strip_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   template<typename T>
   void PrintDefnInput(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     if (d.required)
     {
       std::string name = d.name;
       std::cout << util::CamelCase(name, true) << " " << GetGoType<T>(d);
     }
   }
   
   template<typename T>
   void PrintDefnInput(
       util::ParamData& d,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // param_name *mat.Dense
     if (d.required)
     {
       std::string name = d.name;
       std::cout << util::CamelCase(name, true) << " *" << GetGoType<T>(d);
     }
   }
   
   template<typename T>
   void PrintDefnInput(
       util::ParamData& d,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     // param_name *DataWithInfo
     if (d.required)
     {
       std::string name = d.name;
       std::cout << util::CamelCase(name, true) << " *" << GetGoType<T>(d);
     }
   }
   
   template<typename T>
   void PrintDefnInput(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Get the type names we need to use.
     std::string goStrippedType, strippedType, printedType, defaultsType;
     StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
   
     // param_name *ModelName
     if (d.required)
     {
       std::string name = d.name;
       std::cout << util::CamelCase(name, true) << " *" << goStrippedType;
     }
   }
   
   template<typename T>
   void PrintDefnInput(util::ParamData& d,
                       const void* /* input */,
                       void* /* output */)
   {
     PrintDefnInput<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
