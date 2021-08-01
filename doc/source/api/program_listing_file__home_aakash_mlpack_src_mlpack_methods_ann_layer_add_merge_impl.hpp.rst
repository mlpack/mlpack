
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_merge_impl.hpp:

Program Listing for File add_merge_impl.hpp
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_add_merge_impl.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/add_merge_impl.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP
   #define MLPACK_METHODS_ANN_LAYER_ADD_MERGE_IMPL_HPP
   
   // In case it hasn't yet been included.
   #include "add_merge.hpp"
   
   #include "../visitor/forward_visitor.hpp"
   #include "../visitor/backward_visitor.hpp"
   #include "../visitor/gradient_visitor.hpp"
   
   namespace mlpack {
   namespace ann  {
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   AddMerge<InputDataType, OutputDataType, CustomLayers...>::AddMerge(
       const bool model, const bool run) :
       model(model), run(run), ownsLayers(!model)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   AddMerge<InputDataType, OutputDataType, CustomLayers...>::AddMerge(
       const bool model, const bool run, const bool ownsLayers) :
       model(model), run(run), ownsLayers(ownsLayers)
   {
     // Nothing to do here.
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   AddMerge<InputDataType, OutputDataType, CustomLayers...>::~AddMerge()
   {
     if (!model && ownsLayers)
     {
       std::for_each(network.begin(), network.end(),
           boost::apply_visitor(deleteVisitor));
     }
   }
   
   template <typename InputDataType, typename OutputDataType,
             typename... CustomLayers>
   template<typename InputType, typename OutputType>
   void AddMerge<InputDataType, OutputDataType, CustomLayers...>::Forward(
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
       output += boost::apply_visitor(outputParameterVisitor, network[i]);
     }
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   template<typename eT>
   void AddMerge<InputDataType, OutputDataType, CustomLayers...>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g)
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
   void AddMerge<InputDataType, OutputDataType, CustomLayers...>::Backward(
       const arma::Mat<eT>& /* input */,
       const arma::Mat<eT>& gy,
       arma::Mat<eT>& g,
       const size_t index)
   {
     boost::apply_visitor(BackwardVisitor(boost::apply_visitor(
         outputParameterVisitor, network[index]), gy,
         boost::apply_visitor(deltaVisitor, network[index])), network[index]);
     g = boost::apply_visitor(deltaVisitor, network[index]);
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   template<typename eT>
   void AddMerge<InputDataType, OutputDataType, CustomLayers...>::Gradient(
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
   template<typename eT>
   void AddMerge<InputDataType, OutputDataType, CustomLayers...>::Gradient(
       const arma::Mat<eT>& input,
       const arma::Mat<eT>& error,
       arma::Mat<eT>& /* gradient */,
       const size_t index)
   {
     boost::apply_visitor(GradientVisitor(input, error), network[index]);
   }
   
   template<typename InputDataType, typename OutputDataType,
            typename... CustomLayers>
   template<typename Archive>
   void AddMerge<InputDataType, OutputDataType, CustomLayers...>::serialize(
       Archive& ar, const uint32_t /* version */)
   {
     // Be sure to clear other layers before loading.
     if (cereal::is_loading<Archive>())
       network.clear();
   
     ar(CEREAL_VECTOR_VARIANT_POINTER(network));
     ar(CEREAL_NVP(model));
     ar(CEREAL_NVP(run));
     ar(CEREAL_NVP(ownsLayers));
   }
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
