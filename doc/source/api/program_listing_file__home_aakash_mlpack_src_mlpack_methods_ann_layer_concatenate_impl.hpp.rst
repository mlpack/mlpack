
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concatenate_impl.hpp:

Program Listing for File concatenate_impl.hpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_concatenate_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/concatenate_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_CONCATENATE_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_CONCATENATE_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "concatenate.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType>
   Concatenate<InputDataType, OutputDataType>::Concatenate() :
       inRows(0)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   Concatenate<InputDataType, OutputDataType>::Concatenate(
       const Concatenate& layer) :
       inRows(layer.inRows),
       weights(layer.weights),
       delta(layer.delta),
       concat(layer.concat)
   {
     // Nothing to to here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   Concatenate<InputDataType, OutputDataType>::Concatenate(Concatenate&& layer) :
       inRows(layer.inRows),
       weights(std::move(layer.weights)),
       delta(std::move(layer.delta)),
       concat(std::move(layer.concat))
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType>
   Concatenate<InputDataType, OutputDataType>&
   Concatenate<InputDataType, OutputDataType>::
   operator=(const Concatenate& layer)
   {
     if (this != &layer)
     {
       inRows = layer.inRows;
       weights = layer.weights;
       delta = layer.delta;
       concat = layer.concat;
     }
   
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType>
   Concatenate<InputDataType, OutputDataType>&
   Concatenate<InputDataType, OutputDataType>::
   operator=(Concatenate&& layer)
   {
     if (this != &layer)
     {
       inRows = layer.inRows;
       weights = std::move(layer.weights);
       delta = std::move(layer.delta);
       concat = std::move(layer.concat);
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Concatenate<InputDataType, OutputDataType>::Forward(
       const arma::Mat<eT>& input, arma::Mat<eT>& output)
   {
     if (concat.is_empty())
       Log::Warn << "The concat matrix has not been provided." << std::endl;
   
     if (input.n_cols != concat.n_cols)
     {
       Log::Fatal << "The number of columns of the concat matrix should be equal "
           << "to the number of columns of input matrix." << std::endl;
     }
   
     inRows = input.n_rows;
     output = arma::join_cols(input, concat);
   }
   
   template<typename InputDataType, typename OutputDataType>
   template<typename eT>
   void Concatenate<InputDataType, OutputDataType>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
   {
     g = gy.submat(0, 0, inRows - 1, concat.n_cols - 1);
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
