
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_simple_dqn.hpp:

Program Listing for File simple_dqn.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_simple_dqn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_SIMPLE_DQN_HPP
   #define MLPACK_METHODS_RL_SIMPLE_DQN_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
   
   namespace mlpack {
   namespace rl {
   
   using namespace mlpack::ann;
   
   template<
     typename OutputLayerType = MeanSquaredError<>,
     typename InitType = GaussianInitialization,
     typename NetworkType = FFN<OutputLayerType, InitType>
   >
   class SimpleDQN
   {
    public:
     SimpleDQN() : network(), isNoisy(false)
     { /* Nothing to do here. */ }
   
     SimpleDQN(const int inputDim,
               const int h1,
               const int h2,
               const int outputDim,
               const bool isNoisy = false,
               InitType init = InitType(),
               OutputLayerType outputLayer = OutputLayerType()):
         network(outputLayer, init),
         isNoisy(isNoisy)
     {
       network.Add(new Linear<>(inputDim, h1));
       network.Add(new ReLULayer<>());
       if (isNoisy)
       {
         noisyLayerIndex.push_back(network.Model().size());
         network.Add(new NoisyLinear<>(h1, h2));
         network.Add(new ReLULayer<>());
         noisyLayerIndex.push_back(network.Model().size());
         network.Add(new NoisyLinear<>(h2, outputDim));
       }
       else
       {
         network.Add(new Linear<>(h1, h2));
         network.Add(new ReLULayer<>());
         network.Add(new Linear<>(h2, outputDim));
       }
     }
   
     SimpleDQN(NetworkType& network, const bool isNoisy = false):
         network(network),
         isNoisy(isNoisy)
     { /* Nothing to do here. */ }
   
     void Predict(const arma::mat state, arma::mat& actionValue)
     {
       network.Predict(state, actionValue);
     }
   
     void Forward(const arma::mat state, arma::mat& target)
     {
       network.Forward(state, target);
     }
   
     void ResetParameters()
     {
       network.ResetParameters();
     }
   
     void ResetNoise()
     {
       for (size_t i = 0; i < noisyLayerIndex.size(); i++)
       {
         boost::get<NoisyLinear<>*>
             (network.Model()[noisyLayerIndex[i]])->ResetNoise();
       }
     }
   
     const arma::mat& Parameters() const { return network.Parameters(); }
     arma::mat& Parameters() { return network.Parameters(); }
   
     void Backward(const arma::mat state, arma::mat& target, arma::mat& gradient)
     {
       network.Backward(state, target, gradient);
     }
   
    private:
     NetworkType network;
   
     bool isNoisy;
   
     std::vector<size_t> noisyLayerIndex;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
