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
#include <mlpack/core/optimizers/cne/cne.hpp>

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
 *
 * @tparam OptimizerType The algorithm used among General matrix factorization,
 * Multi layer perceptron and Neural matrix factorization as part of performing
 * Neural Collaborative Filtering.
 */
template<typename OptimizerType>
class NCF
{
 public:
  /**
   * Initialize the NCF object using any algorithm and optimizer, according to
   * which a model will be trained. There are parameters that can
   * be set; default values are provided for each of them.
   *
   * The provided dataset can shall be a coordinate list; that is, a 3-row
   * matrix where each column corresponds to a (user, item, rating) entry in the
   * matrix.
   *
   * @param dataset Data matrix: dense matrix (coordinate lists).
   * @param algorithm Algorithm to be used.
   * @param optimizer Optimizer to be used to train the model.
   * @param embedSize Size of embedding for each user and item being considered.
   * @param neg Number of negative instances to consider per positive instance.
   * @param epochs Number of epochs to train the model on.
   */
  NCF(arma::mat& dataset,
      std::string algorithm,
      OptimizerType& optimizer = OptimizerType(),
      const size_t embedSize = 8,
      const size_t neg = 4,
      const size_t epochs = 100);

  /**
   * To be run once to create a vector of items which haven't been rated by
   * a user. This is used later to create training instances which contain
   * negative instances too.
   *
   * @param numUsers Number of users being considered in the dataset.
   * @param dataset Data matrix: dense matrix (coordinate lists).
   * @param negatives A vector storing unrated items for each user.
   */
  void FindNegatives(arma::mat& dataset);

  /**
   * To be used to get training instance for each epoch. Each training instance
   * will create vectors of users items and labels for training.
   *
   * @param dataset Data matrix: dense matrix (coordinate lists).
   * @param users Matrix to store training data for users.
   * @param items Matrix to store training data for items.
   * @param labels Matrix to store labels (1 for rated and 0 for unrated).
   * @param negatives A vector storing unrated items for each user.
   */
  void GetTrainingInstance(arma::mat& dataset,
                           arma::mat& predictors,
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
   * @param deterministic Whether or not to train or test the model. Note some
   *        layer act differently in training or testing mode.
   */
  double Evaluate(const arma::mat& parameters,
                  const size_t begin,
                  const size_t batchSize,
                  const bool deterministic);

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
  /**
   * Train the model using the specified algorithm and optimizer.
   *
   * @param dataset Data matrix: dense matrix (coordinate lists).
   * @param algorithm Algorithm to be used.
   */
  void Train(arma::mat& dataset, std::string algorithm);

  /**
   * Create the model for General Matrix Factorization.
   *
   * @param data Vector with user and item training data concatenated.
   * @param embedSize Size of embedding for each user and item being considered.
   */
  void CreateGMF(arma::mat& data);

  /**
   * Create the model for Multi Layer Perceptron.
   *
   * @param data Vector with user and item training data concatenated.
   * @param embedSize Size of embedding for each user and item being considered.
   */
  void CreateMLP(arma::mat& data);

  /**
   * Create the model for Neural Matrix Factorization.
   *
   * @param data Vector with user and item training data concatenated.
   * @param embedSize Size of embedding for each user and item being considered.
   */
  void CreateNeuMF(arma::mat& data);

  /**
   * Evaluate the model.
   *
   */
  void EvaluateModel(arma::mat& testData,
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

  const size_t Neg() const { return neg; }

  const size_t EmbedSize() const { return embedSize; }

  /**
   * Serialize the NCF model to the given archive.
   */
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Model for the algorithm to be used.
  FFN<NegativeLogLikelihood<>, RandomInitialization> network;

  //! Optimizer to be used for training.
  OptimizerType optimizer;

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
}; // class NCF

} // namespace cf
} // namespace mlpack

// Include implementation of functions.
#include "ncf_impl.hpp"

#endif
