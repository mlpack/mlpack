
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_categorical_dqn.hpp:

Program Listing for File categorical_dqn.hpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_categorical_dqn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/q_networks/categorical_dqn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_CATEGORICAL_DQN_HPP
   #define MLPACK_METHODS_RL_CATEGORICAL_DQN_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
   #include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
   #include "../training_config.hpp"
   
   namespace mlpack {
   namespace rl {
   
   using namespace mlpack::ann;
   
   template<
     typename OutputLayerType = EmptyLoss<>,
     typename InitType = GaussianInitialization,
     typename NetworkType = FFN<OutputLayerType, InitType>
   >
   class CategoricalDQN
   {
    public:
     CategoricalDQN() :
         network(), atomSize(0), vMin(0.0), vMax(0.0), isNoisy(false)
     { /* Nothing to do here. */ }
   
     CategoricalDQN(const int inputDim,
                    const int h1,
                    const int h2,
                    const int outputDim,
                    TrainingConfig config,
                    const bool isNoisy = false,
                    InitType init = InitType(),
                    OutputLayerType outputLayer = OutputLayerType()):
         network(outputLayer, init),
         atomSize(config.AtomSize()),
         vMin(config.VMin()),
         vMax(config.VMax()),
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
         network.Add(new NoisyLinear<>(h2, outputDim * atomSize));
       }
       else
       {
         network.Add(new Linear<>(h1, h2));
         network.Add(new ReLULayer<>());
         network.Add(new Linear<>(h2, outputDim * atomSize));
       }
     }
   
     CategoricalDQN(NetworkType& network,
                    TrainingConfig config,
                    const bool isNoisy = false):
         network(std::move(network)),
         atomSize(config.AtomSize()),
         vMin(config.VMin()),
         vMax(config.VMax()),
         isNoisy(isNoisy)
     { /* Nothing to do here. */ }
   
     void Predict(const arma::mat state, arma::mat& actionValue)
     {
       arma::mat q_atoms;
       network.Predict(state, q_atoms);
       activations.copy_size(q_atoms);
       actionValue.set_size(q_atoms.n_rows / atomSize, q_atoms.n_cols);
       arma::rowvec support = arma::linspace<arma::rowvec>(vMin, vMax, atomSize);
       for (size_t i = 0; i < q_atoms.n_rows; i += atomSize)
       {
         arma::mat activation = activations.rows(i, i + atomSize - 1);
         arma::mat input = q_atoms.rows(i, i + atomSize - 1);
         softMax.Forward(input, activation);
         activations.rows(i, i + atomSize - 1) = activation;
         actionValue.row(i/atomSize) = support * activation;
       }
     }
   
     void Forward(const arma::mat state, arma::mat& dist)
     {
       arma::mat q_atoms;
       network.Forward(state, q_atoms);
       activations.copy_size(q_atoms);
       for (size_t i = 0; i < q_atoms.n_rows; i += atomSize)
       {
         arma::mat activation = activations.rows(i, i + atomSize - 1);
         arma::mat input = q_atoms.rows(i, i + atomSize - 1);
         softMax.Forward(input, activation);
         activations.rows(i, i + atomSize - 1) = activation;
       }
       dist = activations;
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
   
     void Backward(const arma::mat state,
                   arma::mat& lossGradients,
                   arma::mat& gradient)
     {
       arma::mat activationGradients(arma::size(activations));
       for (size_t i = 0; i < activations.n_rows; i += atomSize)
       {
         arma::mat activationGrad;
         arma::mat lossGrad = lossGradients.rows(i, i + atomSize - 1);
         arma::mat activation = activations.rows(i, i + atomSize - 1);
         softMax.Backward(activation, lossGrad, activationGrad);
         activationGradients.rows(i, i + atomSize - 1) = activationGrad;
       }
       network.Backward(state, activationGradients, gradient);
     }
   
    private:
     NetworkType network;
   
     size_t atomSize;
   
     double vMin;
   
     double vMax;
   
     bool isNoisy;
   
     std::vector<size_t> noisyLayerIndex;
   
     Softmax<> softMax;
   
     arma::mat activations;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
