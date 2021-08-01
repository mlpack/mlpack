
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_ann_layer_base_layer.hpp:

Program Listing for File base_layer.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_ann_layer_base_layer.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/ann/layer/base_layer.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP
   #define MLPACK_METHODS_ANN_LAYER_BASE_LAYER_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/activation_functions/logistic_function.hpp>
   #include <mlpack/methods/ann/activation_functions/identity_function.hpp>
   #include <mlpack/methods/ann/activation_functions/rectifier_function.hpp>
   #include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
   #include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
   #include <mlpack/methods/ann/activation_functions/hard_sigmoid_function.hpp>
   #include <mlpack/methods/ann/activation_functions/swish_function.hpp>
   #include <mlpack/methods/ann/activation_functions/mish_function.hpp>
   #include <mlpack/methods/ann/activation_functions/lisht_function.hpp>
   #include <mlpack/methods/ann/activation_functions/gelu_function.hpp>
   #include <mlpack/methods/ann/activation_functions/elliot_function.hpp>
   #include <mlpack/methods/ann/activation_functions/elish_function.hpp>
   #include <mlpack/methods/ann/activation_functions/gaussian_function.hpp>
   #include <mlpack/methods/ann/activation_functions/hard_swish_function.hpp>
   #include <mlpack/methods/ann/activation_functions/tanh_exponential_function.hpp>
   #include <mlpack/methods/ann/activation_functions/silu_function.hpp>
   
   namespace mlpack {
   namespace ann  {
   
   template <
       class ActivationFunction = LogisticFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   class BaseLayer
   {
    public:
     BaseLayer()
     {
       // Nothing to do here.
     }
   
     template<typename InputType, typename OutputType>
     void Forward(const InputType& input, OutputType& output)
     {
       ActivationFunction::Fn(input, output);
     }
   
     template<typename eT>
     void Backward(const arma::Mat<eT>& input,
                   const arma::Mat<eT>& gy,
                   arma::Mat<eT>& g)
     {
       arma::Mat<eT> derivative;
       ActivationFunction::Deriv(input, derivative);
       g = gy % derivative;
     }
   
     OutputDataType const& OutputParameter() const { return outputParameter; }
     OutputDataType& OutputParameter() { return outputParameter; }
   
     OutputDataType const& Delta() const { return delta; }
     OutputDataType& Delta() { return delta; }
   
     template<typename Archive>
     void serialize(Archive& /* ar */, const uint32_t /* version */)
     {
       /* Nothing to do here */
     }
   
    private:
     OutputDataType delta;
   
     OutputDataType outputParameter;
   }; // class BaseLayer
   
   // Convenience typedefs.
   
   template <
       class ActivationFunction = LogisticFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using SigmoidLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = IdentityFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using IdentityLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = RectifierFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using ReLULayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = TanhFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using TanHLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = SoftplusFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using SoftPlusLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = HardSigmoidFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using HardSigmoidLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = SwishFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using SwishFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = MishFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using MishFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = LiSHTFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using LiSHTFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = GELUFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using GELUFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = ElliotFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using ElliotFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = ElishFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using ElishFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = GaussianFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using GaussianFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = HardSwishFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using HardSwishFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = TanhExpFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using TanhExpFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType>;
   
   template <
       class ActivationFunction = SILUFunction,
       typename InputDataType = arma::mat,
       typename OutputDataType = arma::mat
   >
   using SILUFunctionLayer = BaseLayer<
       ActivationFunction, InputDataType, OutputDataType
   >;
   
   } // namespace ann
   } // namespace mlpack
   
   #endif
