
.. _program_listing_file__home_aakash_mlpack_src_mlpack_bindings_cli_in_place_copy.hpp:

Program Listing for File in_place_copy.hpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_bindings_cli_in_place_copy.hpp>` (``/home/aakash/mlpack/src/mlpack/bindings/cli/in_place_copy.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_BINDINGS_CLI_IN_PLACE_COPY_HPP
   #define MLPACK_BINDINGS_CLI_IN_PLACE_COPY_HPP
   
   #include <mlpack/prereqs.hpp>
   
   namespace mlpack {
   namespace bindings {
   namespace cli {
   
   template<typename T>
   void InPlaceCopyInternal(
       util::ParamData& /* d */,
       util::ParamData& /* input */,
       const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
       const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
       const typename boost::disable_if<std::is_same<T,
           std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
   {
     // Nothing to do.
   }
   
   template<typename T>
   void InPlaceCopyInternal(
       util::ParamData& d,
       util::ParamData& input,
       const typename std::enable_if<
           arma::is_arma_type<T>::value ||
           std::is_same<T,
                        std::tuple<mlpack::data::DatasetInfo, arma::mat>>::value
                    >::type* = 0)
   {
     // Make the output filename the same as the input filename.
     typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
     TupleType& tuple = *boost::any_cast<TupleType>(&d.value);
     std::string& value = std::get<0>(std::get<1>(tuple));
   
     const TupleType& inputTuple = *boost::any_cast<TupleType>(&input.value);
     value = std::get<0>(std::get<1>(inputTuple));
   }
   
   template<typename T>
   void InPlaceCopyInternal(
       util::ParamData& d,
       util::ParamData& input,
       const typename std::enable_if<
           data::HasSerialize<T>::value>::type* = 0)
   {
     // Make the output filename the same as the input filename.
     typedef std::tuple<T*, typename ParameterType<T>::type> TupleType;
     TupleType& tuple = *boost::any_cast<TupleType>(&d.value);
     std::string& value = std::get<1>(tuple);
   
     const TupleType& inputTuple = *boost::any_cast<TupleType>(&input.value);
     value = std::get<1>(inputTuple);
   }
   
   template<typename T>
   void InPlaceCopy(util::ParamData& d,
                    const void* input,
                    void* /* output */)
   {
     // Cast to the correct type.
     InPlaceCopyInternal<typename std::remove_pointer<T>::type>(
         const_cast<util::ParamData&>(d), *((util::ParamData*) input));
   }
   
   } // namespace cli
   } // namespace bindings
   } // namespace mlpack
   
   #endif
