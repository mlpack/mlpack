
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_output_processing.hpp:

Program Listing for File print_output_processing.hpp
====================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_output_processing.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_output_processing.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_OUTPUT_PROCESSING_HPP
   #define MLPACK_BINDINGS_GO_PRINT_OUTPUT_PROCESSING_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "get_type.hpp"
   #include "strip_type.hpp"
   #include <mlpack/bindings/util/camel_case.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string name = d.name;
     name = util::CamelCase(name, true);
     std::cout << prefix << name << " := getParam" << GetType<T>(d)
               << "(\"" << d.name << "\")" << std::endl;
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0,
       const typename std::enable_if<!std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string name = d.name;
     name = util::CamelCase(name, true);
     std::cout << prefix << "var " << name << "Ptr mlpackArma" << std::endl;
     std::cout << prefix << name << " := " << name
               << "Ptr.armaToGonum" << GetType<T>(d)
               << "(\""  << d.name << "\")" << std::endl;
   }
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string name = d.name;
     name = util::CamelCase(name, true);
     std::cout << prefix << "var " << name << "Ptr mlpackArma" << std::endl;
     std::cout << prefix << name << " := " << name << "Ptr.armaToGonumWith"
               << "Info(\""  << d.name << "\")" << std::endl;
   }
   
   template<typename T>
   void PrintOutputProcessing(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Get the type names we need to use.
     std::string goStrippedType, strippedType, printedType, defaultsType;
     StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
   
     const std::string prefix(indent, ' ');
   
     std::string name = d.name;
     name = util::CamelCase(name, true);
     std::cout << prefix << "var " << name << " " << goStrippedType << std::endl;
     std::cout << prefix << name << ".get" << strippedType
               << "(\"" << d.name << "\")" << std::endl;
   }
   
   template<typename T>
   void PrintOutputProcessing(util::ParamData& d,
                              const void* /*input*/,
                              void* /* output */)
   {
     PrintOutputProcessing<typename std::remove_pointer<T>::type>(d, 2);
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
