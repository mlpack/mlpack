
.. _program_listing_file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_dueling_dqn.hpp:

Program Listing for File dueling_dqn.hpp
========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_methods_reinforcement_learning_q_networks_dueling_dqn.hpp>` (``/home/aakash/mlpack/src/mlpack/methods/reinforcement_learning/q_networks/dueling_dqn.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MLPACK_METHODS_RL_DUELING_DQN_HPP
   #define MLPACK_METHODS_RL_DUELING_DQN_HPP
   
   #include <mlpack/prereqs.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
   #include <mlpack/methods/ann/loss_functions/empty_loss.hpp>
   
   namespace mlpack {
   namespace rl {
   
   using namespace mlpack::ann;
   
   template <
     typename OutputLayerType = EmptyLoss<>,
     typename InitType = GaussianInitialization,
     typename CompleteNetworkType = FFN<OutputLayerType, InitType>,
     typename FeatureNetworkType = Sequential<>,
     typename AdvantageNetworkType = Sequential<>,
     typename ValueNetworkType = Sequential<>
   >
   class DuelingDQN
   {
    public:
     DuelingDQN() : isNoisy(false)
     {
       featureNetwork = new Sequential<>();
       valueNetwork = new Sequential<>();
       advantageNetwork = new Sequential<>();
       concat = new Concat<>(true);
   
       concat->Add(valueNetwork);
       concat->Add(advantageNetwork);
       completeNetwork.Add(new IdentityLayer<>());
       completeNetwork.Add(featureNetwork);
       completeNetwork.Add(concat);
     }
   
     DuelingDQN(const int inputDim,
                const int h1,
                const int h2,
                const int outputDim,
                const bool isNoisy = false,
                InitType init = InitType(),
                OutputLayerType outputLayer = OutputLayerType()):
         completeNetwork(outputLayer, init),
         isNoisy(isNoisy)
     {
       featureNetwork = new Sequential<>();
       featureNetwork->Add(new Linear<>(inputDim, h1));
       featureNetwork->Add(new ReLULayer<>());
   
       valueNetwork = new Sequential<>();
       advantageNetwork = new Sequential<>();
   
       if (isNoisy)
       {
         noisyLayerIndex.push_back(valueNetwork->Model().size());
         valueNetwork->Add(new NoisyLinear<>(h1, h2));
         advantageNetwork->Add(new NoisyLinear<>(h1, h2));
   
         valueNetwork->Add(new ReLULayer<>());
         advantageNetwork->Add(new ReLULayer<>());
   
         noisyLayerIndex.push_back(valueNetwork->Model().size());
         valueNetwork->Add(new NoisyLinear<>(h2, 1));
         advantageNetwork->Add(new NoisyLinear<>(h2, outputDim));
       }
       else
       {
         valueNetwork->Add(new Linear<>(h1, h2));
         valueNetwork->Add(new ReLULayer<>());
         valueNetwork->Add(new Linear<>(h2, 1));
   
         advantageNetwork->Add(new Linear<>(h1, h2));
         advantageNetwork->Add(new ReLULayer<>());
         advantageNetwork->Add(new Linear<>(h2, outputDim));
       }
   
       concat = new Concat<>(true);
       concat->Add(valueNetwork);
       concat->Add(advantageNetwork);
   
       completeNetwork.Add(new IdentityLayer<>());
       completeNetwork.Add(featureNetwork);
       completeNetwork.Add(concat);
       this->ResetParameters();
     }
   
     DuelingDQN(FeatureNetworkType& featureNetwork,
                AdvantageNetworkType& advantageNetwork,
                ValueNetworkType& valueNetwork,
                const bool isNoisy = false):
         featureNetwork(featureNetwork),
         advantageNetwork(advantageNetwork),
         valueNetwork(valueNetwork),
         isNoisy(isNoisy)
     {
       concat = new Concat<>(true);
       concat->Add(valueNetwork);
       concat->Add(advantageNetwork);
       completeNetwork.Add(new IdentityLayer<>());
       completeNetwork.Add(featureNetwork);
       completeNetwork.Add(concat);
       this->ResetParameters();
     }
   
     DuelingDQN(const DuelingDQN& /* model */) : isNoisy(false)
     { /* Nothing to do here. */ }
   
     void operator = (const DuelingDQN& model)
     {
       *valueNetwork = *model.valueNetwork;
       *advantageNetwork = *model.advantageNetwork;
       *featureNetwork = *model.featureNetwork;
       isNoisy = model.isNoisy;
       noisyLayerIndex = model.noisyLayerIndex;
     }
   
     void Predict(const arma::mat state, arma::mat& actionValue)
     {
       arma::mat advantage, value, networkOutput;
       completeNetwork.Predict(state, networkOutput);
       value = networkOutput.row(0);
       advantage = networkOutput.rows(1, networkOutput.n_rows - 1);
       actionValue = advantage.each_row() +
           (value - arma::mean(advantage));
     }
   
     void Forward(const arma::mat state, arma::mat& actionValue)
     {
       arma::mat advantage, value, networkOutput;
       completeNetwork.Forward(state, networkOutput);
       value = networkOutput.row(0);
       advantage = networkOutput.rows(1, networkOutput.n_rows - 1);
       actionValue = advantage.each_row() +
           (value - arma::mean(advantage));
       this->actionValues = actionValue;
     }
   
     void Backward(const arma::mat state, arma::mat& target, arma::mat& gradient)
     {
       arma::mat gradLoss;
       lossFunction.Backward(this->actionValues, target, gradLoss);
   
       arma::mat gradValue = arma::sum(gradLoss);
       arma::mat gradAdvantage = gradLoss.each_row() - arma::mean(gradLoss);
   
       arma::mat grad = arma::join_cols(gradValue, gradAdvantage);
       completeNetwork.Backward(state, grad, gradient);
     }
   
     void ResetParameters()
     {
       completeNetwork.ResetParameters();
     }
   
     void ResetNoise()
     {
       for (size_t i = 0; i < noisyLayerIndex.size(); i++)
       {
         boost::get<NoisyLinear<>*>
             (valueNetwork->Model()[noisyLayerIndex[i]])->ResetNoise();
         boost::get<NoisyLinear<>*>
             (advantageNetwork->Model()[noisyLayerIndex[i]])->ResetNoise();
       }
     }
   
     const arma::mat& Parameters() const { return completeNetwork.Parameters(); }
     arma::mat& Parameters() { return completeNetwork.Parameters(); }
   
    private:
     CompleteNetworkType completeNetwork;
   
     Concat<>* concat;
   
     FeatureNetworkType* featureNetwork;
   
     AdvantageNetworkType* advantageNetwork;
   
     ValueNetworkType* valueNetwork;
   
     bool isNoisy;
   
     std::vector<size_t> noisyLayerIndex;
   
     arma::mat actionValues;
   
     MeanSquaredError<> lossFunction;
   };
   
   } // namespace rl
   } // namespace mlpack
   
   #endif
