
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_set_param.hpp:

Program Listing for File set_param.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_set_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/set_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_SET_PARAM_HPP
   #define MLPACK_BINDINGS_CLI_SET_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "parameter_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   void SetParam(
       util::ParamData& d,
       const boost::any& value,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0,
       const typename boost::disable_if<std::is_same<T, bool>>::type* = 0)
   {
     // No mapping is needed.
     d.value = value;
   }
   
   template<typename T>
   void SetParam(
       util::ParamData& d,
       const boost::any& /* value */,
       const typename boost::enable_if<std::is_same<T, bool>>::type* = 0)
   {
     // Force set to the value of whether or not this was passed.
     d.value = d.wasPassed;
   }
   
   template<typename T>
   void SetParam(
       util::ParamData& d,
       const boost::any& value,
       const typename std::enable_if<arma::is_arma_type<T>::value ||
                                     std::is_same<T,
           std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
   {
     // We're setting the string filename.
     typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
     TupleType& tuple = *boost::any_cast<TupleType>(&d.value);
     std::get<0>(std::get<1>(tuple)) = boost::any_cast<std::string>(value);
   }
   
   template<typename T>
   void SetParam(
       util::ParamData& d,
       const boost::any& value,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // We're setting the string filename.
     typedef std::tuple<T*, typename ParameterType<T>::type> TupleType;
     TupleType& tuple = *boost::any_cast<TupleType>(&d.value);
     std::get<1>(tuple) = boost::any_cast<std::string>(value);
   }
   
   template<typename T>
   void SetParam(util::ParamData& d, const void* input, void* /* output */)
   {
     SetParam<typename std::remove_pointer<T>::type>(
         const_cast<util::ParamData&>(d), *((boost::any*) input));
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
