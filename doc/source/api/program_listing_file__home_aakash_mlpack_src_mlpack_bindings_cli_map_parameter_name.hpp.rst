
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_map_parameter_name.hpp:

Program Listing for File map_parameter_name.hpp
===============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_map_parameter_name.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/map_parameter_name.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_MAP_PARAMETER_NAME_HPP
   #define MLPACK_BINDINGS_CLI_MAP_PARAMETER_NAME_HPP
   
   #include <mlpack/core/util/param_data.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   std::string MapParameterName(
       const std::string& identifier,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     return identifier;
   }
   
   template<typename T>
   std::string MapParameterName(
       const std::string& identifier,
       const typename boost::enable_if_c<
           arma::is_arma_type<T>::value ||
           std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                      arma::mat>>::value ||
           data::HasSerialize<T>::value>::type* /* junk */ = 0)
   {
     return identifier + "_file";
   }
   
   template<typename T>
   void MapParameterName(util::ParamData& d,
                         const void* /* input */,
                         void* output)
   {
     // Store the mapped name in the output pointer, which is actually a string
     // pointer.
     *((std::string*) output) =
         MapParameterName<typename std::remove_pointer<T>::type>(d.name);
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
