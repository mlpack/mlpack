
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_names.hpp:

Program Listing for File layer_names.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_names.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer_names.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <boost/variant/static_visitor.hpp>
   #include <string>
   
   using namespace mlpack::ann;
   
   class LayerNameVisitor : public boost::static_visitor<std::string>
   {
    public:
     LayerNameVisitor()
     {
     }
   
     std::string LayerString(AdaptiveMaxPooling<> * /*layer*/) const
     {
       return "adaptivemaxpooling";
     }
   
     std::string LayerString(AdaptiveMeanPooling<> * /*layer*/) const
     {
       return "adaptivemeanpooling";
     }
   
     std::string LayerString(AtrousConvolution<>* /*layer*/) const
     {
       return "atrousconvolution";
     }
   
     std::string LayerString(AlphaDropout<>* /*layer*/) const
     {
       return "alphadropout";
     }
   
     std::string LayerString(BatchNorm<>* /*layer*/) const
     {
       return "batchnorm";
     }
   
     std::string LayerString(Constant<>* /*layer*/) const
     {
       return "constant";
     }
   
     std::string LayerString(Convolution<>* /*layer*/) const
     {
       return "convolution";
     }
   
     std::string LayerString(DropConnect<>* /*layer*/) const
     {
       return "dropconnect";
     }
   
     std::string LayerString(Dropout<>* /*layer*/) const
     {
       return "dropout";
     }
   
     std::string LayerString(FlexibleReLU<>* /*layer*/) const
     {
       return "flexiblerelu";
     }
   
     std::string LayerString(LayerNorm<>* /*layer*/) const
     {
       return "layernorm";
     }
   
     std::string LayerString(Linear<>* /*layer*/) const
     {
       return "linear";
     }
   
     std::string LayerString(LinearNoBias<>* /*layer*/) const
     {
       return "linearnobias";
     }
   
     std::string LayerString(NoisyLinear<>* /*layer*/) const
     {
       return "noisylinear";
     }
   
     std::string LayerString(MaxPooling<>* /*layer*/) const
     {
       return "maxpooling";
     }
   
     std::string LayerString(MeanPooling<>* /*layer*/) const
     {
       return "meanpooling";
     }
   
     std::string LayerString(LpPooling<>* /*layer*/) const
     {
       return "lppooling";
     }
   
     std::string LayerString(MultiplyConstant<>* /*layer*/) const
     {
       return "multiplyconstant";
     }
   
     std::string LayerString(ReLULayer<>* /*layer*/) const
     {
       return "relu";
     }
   
     std::string LayerString(TransposedConvolution<>* /*layer*/) const
     {
       return "transposedconvolution";
     }
   
     std::string LayerString(IdentityLayer<>* /*layer*/) const
     {
       return "identity";
     }
   
     std::string LayerString(TanHLayer<>* /*layer*/) const
     {
       return "tanh";
     }
   
     std::string LayerString(ELU<>* /*layer*/) const
     {
       return "elu";
     }
   
     std::string LayerString(HardTanH<>* /*layer*/) const
     {
       return "hardtanh";
     }
   
     std::string LayerString(LeakyReLU<>* /*layer*/) const
     {
       return "leakyrelu";
     }
   
     std::string LayerString(PReLU<>* /*layer*/) const
     {
       return "prelu";
     }
   
     std::string LayerString(SigmoidLayer<>* /*layer*/) const
     {
       return "sigmoid";
     }
   
     std::string LayerString(LogSoftMax<>* /*layer*/) const
     {
       return "logsoftmax";
     }
   
     /*
      * Return the name of the given layer of type LSTM as a string.
      *
      * @param * Given layer of type LSTM.
      * @return The string representation of the layer.
      */
     std::string LayerString(LSTM<>* /*layer*/) const
     {
       return "lstm";
     }
   
     std::string LayerString(CReLU<>* /*layer*/) const
     {
       return "crelu";
     }
   
     std::string LayerString(Highway<>* /*layer*/) const
     {
       return "highway";
     }
   
     std::string LayerString(GRU<>* /*layer*/) const
     {
       return "gru";
     }
   
     std::string LayerString(Glimpse<>* /*layer*/) const
     {
       return "glimpse";
     }
   
     std::string LayerString(FastLSTM<>* /*layer*/) const
     {
       return "fastlstm";
     }
   
     std::string LayerString(WeightNorm<>* /*layer*/) const
     {
       return "weightnorm";
     }
   
     template<typename T>
     std::string LayerString(T* /*layer*/) const
     {
       return "unsupported";
     }
   
     std::string operator()(MoreTypes layer) const
     {
       return layer.apply_visitor(*this);
     }
   
     template<typename LayerType>
     std::string operator()(LayerType* layer) const
     {
       return LayerString(layer);
     }
   };
