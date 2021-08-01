
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_merge_impl.hpp:

Program Listing for File multiply_merge_impl.hpp
================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_multiply_merge_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/multiply_merge_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_MULTIPLY_MERGE_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "multiply_merge.hpp"
   
   #include "../visitor/forward_visitor.hpp"
   #include "../visitor/backward_visitor.hpp"
   #include "../visitor/gradient_visitor.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::MultiplyMerge(
       const bool model, const bool run) :
       model(model), run(run), ownsLayer(!model)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::MultiplyMerge(
       const MultiplyMerge& layer) :
       model(layer.model),
       run(layer.run),
       ownsLayer(layer.ownsLayer),
       network(layer.network),
       weights(layer.weights)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::MultiplyMerge(
       MultiplyMerge&& layer) :
       model(std::move(layer.model)),
       run(std::move(layer.run)),
       ownsLayer(std::move(layer.ownsLayer)),
       network(std::move(layer.network)),
       weights(std::move(layer.weights))
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>&
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::operator=(
       const MultiplyMerge& layer)
   {
     if (this != &layer)
     {
       model = layer.model;
       run = layer.run;
       ownsLayer = layer.ownsLayer;
       network = layer.network;
       weights = layer.weights;
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>&
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::operator=(
       MultiplyMerge&& layer)
   {
     if (this != &layer)
     {
       model = std::move(layer.model);
       run = std::move(layer.run);
       ownsLayer = std::move(layer.ownsLayer);
       network = std::move(layer.network);
       weights = std::move(layer.weights);
     }
     return *this;
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::~MultiplyMerge()
   {
     if (ownsLayer)
     {
       std::for_each(network.begin(), network.end(),
           boost::apply_visitor(deleteVisitor));
     }
   }
   
   template <typename InputDataType, typename OutputDataType,
             typename... CustomLayers>
   template<typename InputType, typename OutputType>
   void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::Forward(
       const InputType& input, OutputType& output)
   {
     if (run)
     {
       for (size_t i = 0; i < network.size(); ++i)
       {
         boost::apply_visitor(ForwardVisitor(input,
             boost::apply_visitor(outputParameterVisitor, network[i])),
             network[i]);
       }
     }
   
     output = boost::apply_visitor(outputParameterVisitor, network.front());
     for (size_t i = 1; i < network.size(); ++i)
     {
       output %= boost::apply_visitor(outputParameterVisitor, network[i]);
     }
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   template<typename eT>
   void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::Backward(
       const arma::Mat<eT>& /* input */, const arma::Mat<eT>& gy, arma::Mat<eT>& g)
   {
     if (run)
     {
       for (size_t i = 0; i < network.size(); ++i)
       {
         boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
             outputParameterVisitor, network[i]), gy,
             boost::apply_visitor(deltaVisitor, network[i])), network[i]);
       }
   
       g = boost::apply_visitor(deltaVisitor, network[0]);
       for (size_t i = 1; i < network.size(); ++i)
       {
         g += boost::apply_visitor(deltaVisitor, network[i]);
       }
     }
     else
       g = gy;
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   template<typename eT>
   void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& /* gradient */ )
   {
     if (run)
     {
       for (size_t i = 0; i < network.size(); ++i)
       {
         boost::apply_visitor(GradientVisitor(input, error), network[i]);
       }
     }
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   template<typename Archive>
   void MultiplyMerge<InputDataType, OutputDataType, CustomLayers...>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     // Be sure to clear other layers before loading.
     if (cereal::is_loading<Archive>())
       network.clear();
   
     ar(CEREAL_VECTOR_VARIANT_POINTER(network));
     ar(CEREAL_NVP(model));
     ar(CEREAL_NVP(run));
     ar(CEREAL_NVP(ownsLayer));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
