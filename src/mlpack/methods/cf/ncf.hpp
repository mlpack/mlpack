/**
 * @file ncf.hpp
 * @author Haritha Nair
 *
 * Implementation of Neural Collaborative Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_NCF_NCF_HPP
#define MLPACK_METHODS_NCF_NCF_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>

namespace mlpack {
namespace cf {
/**
 * This class implements Neural Collaborative Filtering (NCF). 
 *
 * The data matrix is a (user, item, rating) table.  Each column in the matrix
 * should have three rows.  The first represents the user; the second represents
 * the item; and the third represents the rating.  The user and item, while they
 * are in a matrix that holds doubles, should hold integer (or size_t) values.
 * The user and item indices are assumed to start at 0.
 */
class NCF
{
 public:
  // Default constructor for NCF.
  NCF()
  {
    // Nothing to do here.
  }
  /**
   * Initialize the NCF object using any algorithm and optimizer, according to
   * which a model will be trained. There are parameters that can
   * be set; default values are provided for each of them.
   *
   * The provided dataset can shall be a coordinate list; that is, a 3-row
   * matrix where each column corresponds to a (user, item, rating) entry in the
   * matrix.
   * @tparam OptimizerType The optimizer to train the network on.
   * @param dataset Data matrix: dense matrix (coordinate lists).
   * @param algorithm Algorithm to be used.
   * @param optimizer Optimizer to be used to train the model.
   * @param embedSize Size of embedding for each user and item being considered.
   * @param neg Number of negative instances to consider per positive instance.
   * @param epochs Number of epochs to train the model on.
   * @param implicit Whether to convert data to implicit feedback rating form.
   */
  template<typename OptimizerType = mlpack::optimization::SGD<>>
  NCF(arma::mat& dataset,
      std::string algorithm,
      OptimizerType& optimizer = OptimizerType(),
      const size_t embedSize = 8,
      const size_t neg = 4,
      const size_t epochs = 100,
      bool implicit = false);

  /**
   * To be run once to create a vector of items which haven't been rated by
   * a user. This is used later to create training instances which contain
   * negative instances too.
   */
  void FindNegatives();

  /**
   * To be used to get training instance for each epoch. Each training instance
   * will create vectors of users items and labels for training.
   *
   * @param predictors Matrix to store user and item data.
   * @param responses Matrix to store their response or rating of item.
   */
  void GetTrainingInstance(arma::mat& predictors,
                           arma::mat& responses);

   /**
   * Evaluate the feedforward network with the given parameters, but using only
   * one data point. This is useful for optimizers such as SGD, which require a
   * separable objective function.
   *
   * @param parameters Matrix model parameters.
   * @param begin Index of the starting point to use for objective function
   *        evaluation.
   * @param batchSize Number of points to be passed at a time to use for
   *        objective function evaluation.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t begin,
                  const size_t batchSize);

  /**
   * Evaluate the gradient of the feedforward network with the given parameters,
   * and with respect to only one point in the dataset. This is useful for
   * optimizers such as SGD, which require a separable objective function.
   *
   * @param parameters Matrix of the model parameters to be optimized.
   * @param begin Index of the starting point to use for objective function
   *        gradient evaluation.
   * @param gradient Matrix to output gradient into.
   * @param batchSize Number of points to be processed as a batch for objective
   *        function gradient evaluation.
   */
  void Gradient(const arma::mat& parameters,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize);

  void Shuffle()
  {
    // Nothing to do here.
  }
  /**
   * Train the model using the specified algorithm and optimizer.
   * @tparam OptimizerType The optimizer to train the network on.
   */
  template<typename OptimizerType>
  void Train(OptimizerType optimizer);

  /**
   * Create the model for General Matrix Factorization.
   */
  void CreateGMF();

  /**
   * Create the model for Multi Layer Perceptron.
   */
  void CreateMLP();

  /**
   * Create the model for Neural Matrix Factorization.
   */
  void CreateNeuMF();

  /**
   * Evaluate the model.
   *
   */
  void EvaluateModel(arma::mat& testData,
                     size_t& hitRatio,
                     size_t& rmseMean,
                     const size_t numRecs = 10);

  /**
   * Generates the given number of recommendations for all users.
   *
   * @param numRecs Number of Recommendations
   * @param recommendations Matrix to save recommendations into.
   */
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations);

  /**
   * Generates the given number of recommendations for the specified users.
   *
   * @param numRecs Number of Recommendations
   * @param recommendations Matrix to save recommendations
   * @param users Users for which recommendations are to be generated
   */
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations,
                          const arma::Col<size_t>& users);

  //! Sets negative instances size.
  void Neg(const size_t negValue)
  {
    this->neg = negValue;
  }

  //! Get negative instances value.
  size_t Neg() const { return neg; }

  //! Sets embed size.
  void EmbedSize(const size_t embedSizeValue)
  {
    this->embedSize = embedSizeValue;
  }

  //! Get embed size.
  size_t EmbedSize() const { return embedSize; }

  //! Get the Dataset Matrix.
  const arma::mat& Dataset() const { return dataset; }

  //! Return the number of separable functions (the number of predictor points).
  size_t NumFunctions() const { return numFunctions; }

  /**
   * Serialize the NCF model to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Dataset to perform collaborative filtering on.
  arma::mat dataset;

  //! Model for the algorithm to be used.
  ann::FFN<ann::NegativeLogLikelihood<>, ann::RandomInitialization> network;

  //! Number of negative instances per positive instances in training data.
  size_t neg;

  //! Number of training epochs.
  size_t epochs;

  //! Size of embedding for each user and item.
  size_t embedSize;

  //! Number of users in the dataset.
  size_t numUsers;

  //! Number of items in the dataset.
  size_t numItems;

  //! Negatives for each user stored in vector form.
  std::vector<std::vector<double>> negatives;

  //! Whether to convert the ratings as implicit feedback.
  bool implicit;

  size_t numFunctions;
}; // class NCF

} // namespace cf
} // namespace mlpack

// Include implementation of functions.
#include "ncf_impl.hpp"

#endif
