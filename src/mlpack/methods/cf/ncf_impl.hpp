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
}

/**
 * Compute all unrated items for each user.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::FindNegatives(const size_t numUsers,
                        const size_t numItems,
                        arma::mat& dataset,
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
    vec negativeList = linspace<vec> (0, numItems - 1, numItems);
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
FFN& NCF::CreateGMF(arma::mat& data,
                    const size_t numUsers,
                    const size_t numItems)
{
  size_t size = data.n_rows/2;

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

  return network;
}

/**
 * Create a model for MLP.
 */
template<typename AlgorithmType, typename OptimizerType>
FFN& NCF::CreateMLP(arma::mat& data,
                    const size_t numUsers,
                    const size_t numItems)
{
  size_t size = data.n_rows/2;

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

  return network;
}

/**
 * Create a model for Neural Matrix Factorization.
 */
template<typename AlgorithmType, typename OptimizerType>
FFN& NCF::CreateNeuMF(arma::mat& data,
                      const size_t numUsers,
                      const size_t numItems)
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
  // Being implemented.
}

/**
 * Get recommendations for given set of users.
 */
template<typename AlgorithmType, typename OptimizerType>
void NCF::GetRecommendations(const size_t numRecs,
                             arma::Mat<size_t>& recommendations,
                             const arma::Col<size_t>& users);
{
  // Being implemented.
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
}

} // namespace cf
} // namespace mlpack

#endif
