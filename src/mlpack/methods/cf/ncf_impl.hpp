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

#ifndef MLPACK_METHODS_NCF__NCF_IMPL_HPP
#define MLPACK_METHODS_NCF_IMPL_NCF_IMPL_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/ncf.hpp>

namespace mlpack {
namespace cf {

/**
 * Construct the NCF object using the desired algorithm and optimizer.
 */
template<typename AlgorithmType, typename OptimizerType>
NCF::NCF(arma::mat& dataset,
         OptimizerType optimizer,
         const size_t embedSize,
         const size_t neg,
         const size_t epochs):
    optimizer(optimizer),
    embedSize(embedSize),
    neg(neg),
    epochs(epochs)
{
  if (embedSize < 1)
  {
    Log::Warn << "NCF::NCF(): Embedding size should be > 0 ("
        << embedSize << " given). Setting value to 8.\n";
    // Set default value of 8.
    this->embedSize = 8;
  }
  numUsers = (size_t) max(dataset.row(0)) + 1;
  numItems = (size_t) max(dataset.row(1)) + 1;
}

/**
 * Compute all unrated items for each user.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::FindNegatives(arma::mat& dataset,
                        std::vector<std::vector<double>>& negatives)
{
  for (int i = 0; i< numUsers; i++)
  {
    // Find items the user has rated.
    arma::uvec userRates = arma::find(dataset.row(0) == i);
    arma::mat itemRates = dataset.cols(userRates);

    itemRates.shed_row(0);
    itemRates.shed_row(1);

    // List of all items.
    vec negativeList = linspace<vec> (0,3705,3706);
    for (int j = 0; j < itemRates.n_cols; j++)
    {
      // Remove items which have been rated.
      arma::uvec temp = arma::find(negativeList ==  itemRates(j));
      negativeList.shed_row(temp(0));
    }

    // Add all negatives to a vector.
    stdvec negList = conv_to<stdvec>::from(negativeList);
    negatives.push_back(negList);
  }
}

/**
 * Create training instances using both positive and negative instances.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::GetTrainingInstance(arma::mat& dataset,
                              arma::mat& users,
                              arma::mat& items,
                              arma::mat& labels,
                              std::vector<std::vector<double>>& negatives)
{
  long long int q = 0;
  int temp;

  for (int i = 0; i < dataset.n_cols; i++)
  {
    temp = neg;

    // Rating exists.
    users(q) = dataset(0, i);
    items(q) = dataset(1, i);
    labels(q) = 1;
    q++;

    // From find negatives.
    int val = negatives[dataset(0, i)].size();

    while (temp != 0)
    {
      int j = math::RandInt(val);
      // Add negatives.
      users(q) = dataset(0, i);
      items(q) = j;
      labels(q) = 0;
      q++;
      temp--;
    }
  }
}

/**
 * Train the model using the given optimizer.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::Train(arma::mat& dataset)
{
  // Being implemented.
}

/**
 * Create a model for GMF.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::CreateGMF(arma::mat& data, size_t embedSize)
{
  size_t size = data/2;

  // User sub-network.
  Sequential<>* userModel = new Sequential<>();
  userModel->Add<Subview<> >(1, 0, size - 1);
  userModel->Add<Embedding<> >(numUsers, embedSize);
  userModel->Add<Subview<> >(size, 0, embedSize - 1, 0, size - 1);

  // Item sub-network.
  Sequential<>* itemModel = new Sequential<>();
  itemModel->Add<Subview<> >(1, size, data.n_rows - 1);
  itemModel->Add<Embedding<> >(numItems, embedSize);
  userModel->Add<Subview<> >(size, 0, embedSize - 1, 0, size - 1);

  // Merge the user and item sub-network.
  MultiplyMerge<> mergeModel(true, true);
  mergeModel.Add(userModel);
  mergeModel.Add(itemModel);

  // Create the main network.
  FFN<NegativeLogLikelihood<>, RandomInitialization> network;
  network.Add<IdentityLayer<> >();
  network.Add<MultiplyMerge<> >(mergeModel);
  network.Add<SigmoidLayer<> >();
}

/**
 * Create a model for MLP.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::CreateMLP(arma::mat& data, size_t embedSize)
{
  size_t size = data/2;

  // User sub-network.
  Sequential<>* userModel = new Sequential<>();
  userModel->Add<Subview<> >(1, 0, size - 1);
  userModel->Add<Embedding<> >(numUsers, embedSize);

  // Item sub-network.
  Sequential<>* itemModel = new Sequential<>();
  itemModel->Add<Subview<> >(1, size, data.n_rows - 1);
  itemModel->Add<Embedding<> >(numItems, embedSize);

  // Merge the user and item sub-network.
  Concat<>* mergeModel = new Concat<>(true, true);
  mergeModel->Add(userModel);
  mergeModel->Add(itemModel);

  // Create the main network.
  FFN<NegativeLogLikelihood<>, RandomInitialization> network;
  network.Add<IdentityLayer<> >();
  network.Add(mergeModel);
  network.Add<Subview<> >(2, 0, (embedSize * size) - 1, 0, 1);
  network.Add<SigmoidLayer<> >();
}

/**
 * Create a model for Neural Matrix Factorization.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::CreateNeuMF(arma::mat& data, size_t embedSize)
{
  // To be added.
}

/**
 * Evaluate the model.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::Evaluate()
{
  // Being implemented.
}

/**
 * Get recommendations for all users.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::GetRecommendations(const size_t numRecs,
                             arma::Mat<size_t>& recommendations)
{
  // Generate list of users.
  arma::Col<size_t> users = arma::linspace<arma::Col<size_t> >(0,
      numUsers - 1, numUsers);

  // Call the main overload for recommendations.
  GetRecommendations(numRecs, recommendations, users);
}

/**
 * Get recommendations for given set of users.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::GetRecommendations(const size_t numRecs,
                             arma::Mat<size_t>& recommendations,
                             const arma::Col<size_t>& users);
{
  // Column vector of all items.
  arma::Col<size_t> itemVec = arma::linspace<arma::Col<size_t> >(0,
      numItems - 1, numItems);
  arma::Col<size_t> predictors, userVec(numItems);

  // Predict rating for all user item combinations for given users.
  for (size_t i = 0; i < users.n_elem; i++)
  {
    arma::mat results;
    userVec.fill(users[i]);

    // Form input for the network.
    predictors = arma::join_vert(userVec, itemVec);
    network.Predict(predictors, results);

    // Find top k recommendations.
    for (size_t k = 0; k < numRecs; k++)
    {
      recommendations(k, i) = (size_t) arma::index_max(results);
      itemScore(arma::index_max(itemScore)) = 0;
    }
  }
}

/**
 * Serialize the NCF model to the given archive.
 */
template<typename AlgorithmType, typename OptimizerType>
template<typename Archive>
void serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(algorithm);
  ar & BOOST_SERIALIZATION_NVP(optimizer);
  ar & BOOST_SERIALIZATION_NVP(neg);
  ar & BOOST_SERIALIZATION_NVP(epochs);
  ar & BOOST_SERIALIZATION_NVP(embedSize);
  ar & BOOST_SERIALIZATION_NVP(numUsers);
  ar & BOOST_SERIALIZATION_NVP(numItems);
}

} // namespace cf
} // namespace mlpack

#endif
