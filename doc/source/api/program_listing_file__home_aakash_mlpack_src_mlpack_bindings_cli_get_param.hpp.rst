
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_get_param.hpp:

Program Listing for File get_param.hpp
======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_get_param.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/get_param.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_GET_PARAM_HPP
   #define MLPACK_BINDINGS_CLI_GET_PARAM_HPP
   
   #include <mlpack/prereqs.hpp>
   #include "parameter_type.hpp"
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   T& GetParam(
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
   T& GetParam(
       util::ParamData& d,
       const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
   {
     // If the matrix is an input matrix, we have to load the matrix.  'value'
     // contains the filename.  It's possible we could load empty matrices many
     // times, but I am not bothered by that---it shouldn't be something that
     // happens.
     typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
     TupleType& tuple = *boost::any_cast<TupleType>(&d.value);
     const std::string& value = std::get<0>(std::get<1>(tuple));
     T& matrix = std::get<0>(tuple);
     size_t& n_rows = std::get<1>(std::get<1>(tuple));
     size_t& n_cols = std::get<2>(std::get<1>(tuple));
     if (d.input && !d.loaded)
     {
       // Call correct data::Load() function.
       if (arma::is_Row<T>::value || arma::is_Col<T>::value)
         data::Load(value, matrix, true);
       else
         data::Load(value, matrix, true, !d.noTranspose);
       n_rows = matrix.n_rows;
       n_cols = matrix.n_cols;
       d.loaded = true;
     }
   
     return matrix;
   }
   
   template<typename T>
   T& GetParam(
       util::ParamData& d,
       const typename boost::enable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     // If this is an input parameter, we need to load both the matrix and the
     // dataset info.
     typedef std::tuple<T, std::tuple<std::string, size_t, size_t>> TupleType;
     TupleType* tuple = boost::any_cast<TupleType>(&d.value);
     const std::string& value = std::get<0>(std::get<1>(*tuple));
     T& t = std::get<0>(*tuple);
     size_t& n_rows = std::get<1>(std::get<1>(*tuple));
     size_t& n_cols = std::get<2>(std::get<1>(*tuple));
     if (d.input && !d.loaded)
     {
       data::Load(value, std::get<1>(t), std::get<0>(t), true, !d.noTranspose);
       n_rows = std::get<1>(t).n_rows;
       n_cols = std::get<1>(t).n_cols;
       d.loaded = true;
     }
   
     return t;
   }
   
   template<typename T>
   T*& GetParam(
       util::ParamData& d,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
   {
     // If the model is an input model, we have to load it from file.  'value'
     // contains the filename.
     typedef std::tuple<T*, std::string> TupleType;
     TupleType* tuple = boost::any_cast<TupleType>(&d.value);
     const std::string& value = std::get<1>(*tuple);
     if (d.input && !d.loaded)
     {
       T* model = new T();
       data::Load(value, "model", *model, true);
       d.loaded = true;
       std::get<0>(*tuple) = model;
     }
     return std::get<0>(*tuple);
   }
   
   template<typename T>
   void GetParam(util::ParamData& d, const void* /* input */, void* output)
   {
     // Cast to the correct type.
     *((T**) output) = &GetParam<typename std::remove_pointer<T>::type>(d);
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
