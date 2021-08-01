
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_parameter_type.hpp:

Program Listing for File parameter_type.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_parameter_type.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/parameter_type.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_PARAMETER_TYPE_HPP
   #define MLPACK_BINDINGS_CLI_PARAMETER_TYPE_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   // Default: HasSerialize = false.
   template<bool HasSerialize, typename T>
   struct ParameterTypeDeducer
   {
     typedef T type;
   };
   
   // If we have a serialize() function, then the type is a string.
   template<typename T>
   struct ParameterTypeDeducer<true, T>
   {
     typedef std::string type;
   };
   
   template<typename T>
   struct ParameterType
   {
     typedef typename ParameterTypeDeducer<data::HasSerialize<T>::value, T>::type
         type;
   };
   
   template<typename eT>
   struct ParameterType<arma::Col<eT>>
   {
     typedef std::tuple<std::string, size_t, size_t> type;
   };
   
   template<typename eT>
   struct ParameterType<arma::Row<eT>>
   {
     typedef std::tuple<std::string, size_t, size_t> type;
   };
   
   template<typename eT>
   struct ParameterType<arma::Mat<eT>>
   {
     typedef std::tuple<std::string, size_t, size_t> type;
   };
   
   template<typename eT, typename PolicyType>
   struct ParameterType<std::tuple<mlpack::data::DatasetMapper<PolicyType,
                            std::string>, arma::Mat<eT>>>
   {
     typedef std::tuple<std::string, size_t, size_t> type;
   };
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
