/**
 * @file methods/ann/dbn.hpp
 * @author Himanshu Pathak
 *
 * Definition of the DBN class, which implements deep belief neural networks.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_ANN_FFN_HPP
#define MLPACK_METHODS_ANN_FFN_HPP

#include <mlpack/prereqs.hpp>

#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/rbm/rbm.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include "init_rules/network_init.hpp"

#include <ensmallen.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a deep belief network.
 *
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam CustomLayers Any set of custom layers that could be a part of the
 *         feed forward network.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename InitializationRuleType = RandomInitialization,
  typename... CustomLayers
>
class DBN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = DBN<OutputLayerType, InitializationRuleType>;

  /**
   * Create the DBN object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param outputLayer Output layer used to evaluate the network.
   * @param initializeRule Optional instantiated InitializationRule object
   *        for initializing the network parameter.
   */
  DBN(trainData,
      learninRate = 1e-5,
      learninRateDecay = false,
      increaseToCDK = false,
      xavierInit = false);

  //! Copy constructor.
  DBN(const DBN&);

  //! Move constructor.
  DBN(DBN&&);

  //! Copy/move assignment operator.
  DBN& operator = (DBN);

  //! Destructor to release allocated memory.
  ~DBN();

  /**
   * Train the deep belief network on the given input data using the given
   * optimizer.
   *
   * This will use the existing model parameters as a starting point for the
   * optimization. If this is not what you want, then you should access the
   * parameters vector directly with Parameters() and modify it as desired.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @tparam OptimizerType Type of optimizer to use to train the model.
   * @tparam CallbackTypes Types of Callback Functions.
   * @param predictors Input training variables.
   * @param responses Outputs results from input training variables.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(OptimizerType& optimizer,
               CallbackTypes&&... callbacks);

  /**
   * Predict the responses to a given set of predictors. The responses will
   * reflect the output of the given output layer as returned by the
   * output layer function.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param predictors Input predictors.
   * @param results Matrix to put output predictions of responses into.
   */
  void Predict(arma::mat predictors, arma::mat& results);

  /**
   * Shuffle the order of function visitation. This may be called by the
   * optimizer.
   */
  void Shuffle();

  /*
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class RBM, class... Args>
  void Add(Args... args) { network.push_back(new RBM(args...)); }

  /*
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(RBM<GaussianInitialization> layer) { network.push_back(layer); }

  //! Get the network model.
  const std::vector<RBM<GaussianInitialization>>& Model() const
  {
    return network;
  }
  //! Modify the network model.  Be careful!  If you change the structure of the
  //! network or parameters for layers, its state may become invalid, so be sure
  //! to call ResetParameters() afterwards.
  std::vector<RBM<GaussianInitialization> >& Model() { return network; }

  //! Get the matrix of data points (predictors).
  const arma::mat& Predictors() const { return predictors; }
  //! Modify the matrix of data points (predictors).
  arma::mat& Predictors() { return predictors; }

  /**
   * Reset the module infomration (weights/parameters).
   */
  void ResetParameters();

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

  /**
   * Perform the forward pass of the data in real batch mode.
   *
   * Forward and Backward should be used as a pair, and they are designed mainly
   * for advanced users. User should try to use Predict and Train unless those
   * two functions can't satisfy some special requirements.
   *
   * @param inputs The input data.
   * @param results The predicted results.
   */
  template<typename PredictorsType, typename ResponsesType>
  void Forward(const PredictorsType& inputs, ResponsesType& results);

 private:

  /**
   * Swap the content of this network with given network.
   *
   * @param network Desired source network.
   */
  //! Locally-stored model modules.
  std::vector<RBM<GaussianInitialization> > network;

  void Swap(FFN& network);

  arma::mat trainData;

  double learninRate;

  bool learninRateDecay;

  bool increaseToCDK;

  bool xavierInit;
  arma::mat prdictors;
}; // class DBN

} // namespace ann
} // namespace mlpack

//! Set the serialization version of the FFN class.  Multiple template arguments
//! makes this ugly...
namespace boost {
namespace serialization {

template<typename OutputLayerType,
         typename InitializationRuleType,
         typename... CustomLayer>
struct version<
    mlpack::ann::DBN<OutputLayerType, InitializationRuleType, CustomLayer...>>
{
  BOOST_STATIC_CONSTANT(int, value = 2);
};

} // namespace serialization
} // namespace boost

// Include implementation.
#include "dbn_impl.hpp"

#endif