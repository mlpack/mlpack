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

#include <ensmallen.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of a deep belief network.
 *
 * This class can be used to stack multiple RBM and train them layer by using
 * greedy approach.
 *
 * For more information, refer the following paper.
 *
 * @code
 * @article{He15,
 *   author    = {Geoffrey E. Hinton, Simon Osindero},
 *   title     = {A fast learning algorithm for deep belief nets},
 *   year      = {2006},
 *   url       = {http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf},
 *   PMID    = {16764513},
 * }
 * @endcode
 *
 *
 * @tparam InputType The input data type used in the network.
 * @tparam OutputType The output layer type used in the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename InputType = arma::mat,
  typename OutputType = arma::mat,
  typename InitializationRuleType = GaussianInitialization
>
class DBN
{
 public:
  //! Convenience typedef for the internal model construction.
  using NetworkType = DBN<InputType, OutputType, InitializationRuleType>;

  /**
   * Create the DBN object.
   *
   * Optionally, specify which initialize rule and performance function should
   * be used.
   *
   * If you want to pass in a parameter and discard the original parameter
   * object, be sure to use std::move to avoid unnecessary copy.
   *
   * @param predictors Input training data.
   */
  DBN(InputType predictors);

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
   * @param optimizer Instantiated optimizer used to train the model.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(OptimizerType& optimizer,
               CallbackTypes&&... callbacks);
  /**
   * Train single layer of the deep belief network on the given input
   * data using the given optimizer.
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
   * @param layerNumber The number of layer which you want to train range from 0 
   *      to network.size() - 1.
   * @param optimizer Instantiated optimizer used to train the model.
   * @param callbacks Callback function for ensmallen optimizer `OptimizerType`.
   *      See https://www.ensmallen.org/docs.html#callback-documentation.
   * @return The final objective of the trained model (NaN or Inf on error).
   */
  template<typename OptimizerType, typename... CallbackTypes>
  double Train(const double layerNumber,
               OptimizerType& optimizer,
               CallbackTypes&&... callbacks);


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
   * @param layer The RBM Layer to be added to the model.
   */
  void Add(RBM<InitializationRuleType> layer) { network.push_back(layer); }

  //! Get the network model.
  const std::vector<RBM<InitializationRuleType> >& Model() const
  {
    return network;
  }
  //! Modify the network model.  Be careful!  If you change the structure of the
  //! network or parameters for layers, its state may become invalid, so be sure
  //! to call ResetParameters() afterwards.
  std::vector<RBM<InitializationRuleType> >& Model() { return network; }

  //! Get the matrix of data points (predictors).
  const arma::mat& Predictors() const { return predictors; }
  //! Modify the matrix of data points (predictors).
  arma::mat& Predictors() { return predictors; }

  /**
   * Reset Every RBM use as layer.
   */
  void Reset();

  /**
   * Setting Bias of the RBM layers.
   */
  void SetBias();

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
  void Forward(const InputType& inputs, OutputType& results);

 private:
  /**
   * Swap the content of this network with given network.
   *
   * @param network Desired source network.
   */
  //! Locally-stored model modules.
  std::vector<RBM<InitializationRuleType> > network;

  //! Locally stored train data.
  InputType predictors;

  //! Locally stored output by Forward function.
  OutputType results;
}; // class DBN

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "dbn_impl.hpp"

#endif
