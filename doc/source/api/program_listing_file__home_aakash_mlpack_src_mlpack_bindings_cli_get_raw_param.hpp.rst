
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_get_raw_param.hpp:

Program Listing for File get_raw_param.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_get_raw_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/get_raw_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_GET_RAW_PARAM_HPP
   #define MLPACK_BINDINGS_CLI_GET_RAW_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "parameter_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   T& GetRawParam(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     // No mapping is needed, so just cast it directly.
     return *boost::any_cast<T>(&d.value);
   }
   
   template<typename T>
   T& GetRawParam(
       util::ParamData& d,
       const typename boost::enable_if_c<
           arma::is_arma_type<T>::value ||
           std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                      arma::mat>>::value>::type* = 0)
   {
     // Don't load the matrix.
     typedef std::tuple<T, std::tuple<std::string, size_t, size_t>> TupleType;
     T& value = std::get<0>(*boost::any_cast<TupleType>(&d.value));
     return value;
   }
   
   template<typename T>
   T*& GetRawParam(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // Don't load the model.
     typedef std::tuple<T*, std::string> TupleType;
     T*& value = std::get<0>(*boost::any_cast<TupleType>(&d.value));
     return value;
   }
   
   template<typename T>
   void GetRawParam(util::ParamData& d,
                    const void* /* input */,
                    void* output)
   {
     // Cast to the correct type.
     *((T**) output) = &GetRawParam<typename std::remove_pointer<T>::type>(
         const_cast<util::ParamData&>(d));
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
