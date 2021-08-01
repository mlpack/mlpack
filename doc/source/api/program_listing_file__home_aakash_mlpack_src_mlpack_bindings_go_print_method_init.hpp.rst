
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_go_print_method_init.hpp:

Program Listing for File print_method_init.hpp
==============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_go_print_method_init.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/go/print_method_init.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_GO_PRINT_METHOD_INIT_HPP
   #define MLPACK_BINDINGS_GO_PRINT_METHOD_INIT_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "get_go_type.hpp"
   #include "strip_type.hpp"
   #include <mlpack/bindings/util/camel_case.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace go {
   
   template<typename T>
   void PrintMethodInit(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string def = "nil";
     if (std::is_same<T, bool>::value)
       def = "false";
   
     // Capitalize the first letter of parameter name so it is
     // of exported type in Go.
     std::string name = d.name;
     std::string goParamName = name;
     if (!name.empty())
     {
       goParamName = util::CamelCase(goParamName, false);
     }
   
     // Only print param that are not required.
     if (!d.required)
     {
       if (d.cppType == "std::string")
       {
         std::string value = boost::any_cast<std::string>(d.value);
         std::cout << prefix << goParamName << ": \""
                   << value << "\"," << std::endl;
       }
       else if (d.cppType == "double")
       {
         double value = boost::any_cast<double>(d.value);
         std::cout << prefix << goParamName << ": " << value << "," << std::endl;
       }
       else if (d.cppType == "int")
       {
         int value = boost::any_cast<int>(d.value);
         std::cout << prefix << goParamName << ": " << value << "," << std::endl;
       }
       else if (d.cppType == "bool")
       {
         bool value = boost::any_cast<bool>(d.value);
         if (value == 0)
           std::cout << prefix << goParamName << ": false," << std::endl;
         else
           std::cout << prefix << goParamName << ": true," << std::endl;
       }
     }
   }
   
   template<typename T>
   void PrintMethodInit(
       util::ParamData& d,
       const size_t indent,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string def = "nil";
     if (std::is_same<T, bool>::value)
       def = "false";
   
     // Capitalize the first letter of parameter name so it is
     // of exported type in Go.
     std::string name = d.name;
     std::string goParamName = name;
     if (!name.empty())
     {
       goParamName = util::CamelCase(goParamName, false);
     }
   
     // Only print param that are not required.
     if (!d.required)
     {
       std::cout << prefix << goParamName << ": " << def << ","
           << std::endl;
     }
   }
   
   template<typename T>
   void PrintMethodInit(
       util::ParamData& d,
       const size_t indent,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string def = "nil";
     if (std::is_same<T, bool>::value)
       def = "false";
   
     // Capitalize the first letter of parameter name so it is
     // of exported type in Go.
     std::string name = d.name;
     std::string goParamName = name;
     if (!name.empty())
     {
       goParamName = util::CamelCase(goParamName, false);
     }
   
     // Only print param that are not required.
     if (!d.required)
     {
       std::cout << prefix << goParamName << ": " << def << ","
           << std::endl;
     }
   }
   
   template<typename T>
   void PrintMethodInit(
       util::ParamData& d,
       const size_t indent,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     const std::string prefix(indent, ' ');
   
     std::string def = "nil";
     if (std::is_same<T, bool>::value)
       def = "false";
   
     // Capitalize the first letter of parameter name so it is
     // of exported type in Go.
     std::string name = d.name;
     std::string goParamName = name;
     if (!name.empty())
     {
       goParamName = util::CamelCase(goParamName, false);
     }
   
     // Only print param that are not required.
     if (!d.required)
     {
       std::cout << prefix << goParamName << ": " << def << ","
                 << std::endl;
     }
   }
   
   template<typename T>
   void PrintMethodInit(util::ParamData& d,
                        const void* input,
                        void* /* output */)
   {
     PrintMethodInit<typename std::remove_pointer<T>::type>(d,
         *((size_t*) input));
   }
   
   } // namespace go
   } // namespace bindings
   } // namespace mlpack
   
   #endif
