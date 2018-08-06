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

#ifndef MLPACK_METHODS_NCF_NCF_IMPL_HPP
#define MLPACK_METHODS_NCF_NCF_IMPL_HPP

#include <mlpack/prereqs.hpp>
#include "ncf.hpp"

namespace mlpack {
namespace cf {
/**
 * Construct the NCF object using the desired algorithm and optimizer.
 */
template<typename OptimizerType>
NCF::NCF(arma::mat& dataset,
         std::string algorithm,
         OptimizerType& optimizer,
         const size_t embedSize,
         const size_t neg,
         const size_t epochs,
         bool implicit):
    dataset(dataset),
    neg(neg),
    epochs(epochs),
    embedSize(embedSize),
    implicit(implicit)
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

  if (algorithm == "GMF")
  {
    CreateGMF();
  }
  else if (algorithm == "MLP")
  {
    CreateMLP();
  }
  else if (algorithm == "NeuMF")
  {
    CreateNeuMF();
  }

  Train(optimizer);
}

/**
 * Compute all unrated items for each user.
 */
void NCF::FindNegatives()
{
  negatives.clear();

  for (size_t i = 0; i< numUsers; i++)
  {
    // Find items the user has rated.
    arma::uvec userRates = arma::find(dataset.row(0) == i);
    arma::mat itemRates = dataset.cols(userRates);

    itemRates.shed_row(0);
    itemRates.shed_row(1);
    // List of all items.n
    arma::vec negativeList = arma::linspace<arma::vec>(
        0, numItems - 1, numItems);

    for (size_t j = 0; j < itemRates.n_cols; j++)
    {
      // Remove items which have been rated.
      arma::uvec temp = arma::find(negativeList ==  itemRates(j));
      if(temp.n_rows != 0)
        negativeList.shed_row(temp(0));
    }

    std::vector<double> negList;
    // Add all negatives to a vector.
    negList = arma::conv_to<std::vector<double>>::from(negativeList);
    negatives.push_back(negList);
  }
}

/**
 * Create training instances using both positive and negative instances.
 */
void NCF::GetTrainingInstance(arma::mat& predictors,
                              arma::mat& responses)
{
  size_t q = 0, temp;
  arma::colvec users(dataset.n_cols * (neg+1));
  arma::colvec resp(dataset.n_cols * (neg+1)), items(dataset.n_cols * (neg+1));

  for (size_t i = 0; i < dataset.n_cols; i++)
  {
    temp = neg;

    // Rating exists.
    users(q) = dataset(0, i);
    items(q) = dataset(1, i);
    resp(q) = implicit ? 1 : dataset(2, i);
    q++;
    // From find negatives.
    size_t val = negatives[dataset(0, i)].size();

    while (temp != 0)
    {
      size_t j = math::RandInt(val);
      // Add negatives.
      users(q) = dataset(0, i);
      items(q) = j;
      resp(q) = 0;
      q++;
      temp--;
    }
  }
  predictors = arma::join_vert(users, items);
  responses = arma::conv_to<arma::mat>::from(resp);
}

double NCF::Evaluate(const arma::mat& parameters,
                     const size_t begin,
                     const size_t batchSize,
                     const bool deterministic)
{
  double loss = network.Evaluate(parameters, begin, batchSize, deterministic);
  return loss;
}

void NCF::Gradient(const arma::mat& parameters,
                   const size_t begin,
                   arma::mat& gradient,
                   const size_t batchSize)
{
  network.Gradient(parameters, begin, gradient, batchSize);
  GetTrainingInstance(network.Predictors(), network.Responses());
}

/**
 * Train the model using the given optimizer.
 */
template<typename OptimizerType>
void NCF::Train(OptimizerType optimizer)
{
  arma::mat predictors, responses;

  FindNegatives();

  GetTrainingInstance(predictors, responses);

  network.ResetData(std::move(predictors), std::move(responses));

  // Train the model.
  Timer::Start("ncf_optimization");
  const double out = optimizer.Optimize(this->network, network.Parameters());
  Timer::Stop("ncf_optimization");

  Log::Info << "NCF::NCF(): final objective of trained model is " << out
      << "." << std::endl;
}

/**
 * Create a model for GMF.
 */
void NCF::CreateGMF()
{
  size_t size = dataset.n_cols*(neg+1);

  // User sub-network.
  ann::Sequential<>* userModel = new ann::Sequential<>();
  userModel->Add<ann::Subview<> >(1, 0, size - 1);
  userModel->Add<ann::Embedding<> >(numUsers, embedSize);
  userModel->Add<ann::Subview<> >(size, 0, embedSize - 1, 0, size - 1);

  // Item sub-network.
  ann::Sequential<>* itemModel = new ann::Sequential<>();
  itemModel->Add<ann::Subview<> >(1, size, (size * 2) - 1);
  itemModel->Add<ann::Embedding<> >(numItems, embedSize);
  itemModel->Add<ann::Subview<> >(size, 0, embedSize - 1, 0, size - 1);

  // Merge the user and item sub-network.
  ann::MultiplyMerge<> mergeModel(true, true);
  mergeModel.Add(userModel);
  mergeModel.Add(itemModel);

  // Create the main network.
  network.Add<ann::IdentityLayer<> >();
  network.Add<ann::MultiplyMerge<> >(mergeModel);
  network.Add<ann::SigmoidLayer<> >();
}

/**
 * Create a model for MLP.
 */
void NCF::CreateMLP()
{
  size_t size = dataset.n_cols*(neg+1);

  // User sub-network.
  ann::Sequential<>* userModel = new ann::Sequential<>();
  userModel->Add<ann::Subview<> >(1, 0, size - 1);
  userModel->Add<ann::Embedding<> >(numUsers, embedSize);

  // Item sub-network.
  ann::Sequential<>* itemModel = new ann::Sequential<>();
  itemModel->Add<ann::Subview<> >(1, size, (size * 2) - 1);
  itemModel->Add<ann::Embedding<> >(numItems, embedSize);

  // Merge the user and item sub-network.
  ann::Concat<>* mergeModel = new ann::Concat<>(true, true);
  mergeModel->Add(userModel);
  mergeModel->Add(itemModel);

  // Create the main network.
  network.Add<ann::IdentityLayer<> >();
  network.Add(mergeModel);
  network.Add<ann::Subview<> >(2, 0, (embedSize * size) - 1, 0, 1);
  network.Add<ann::SigmoidLayer<> >();
}

/**
 * Create a model for Neural Matrix Factorization.
 */
void NCF::CreateNeuMF()
{
  // Being debugged.
}

/**
 * Evaluate the model.
 */
void NCF::EvaluateModel(arma::mat& testData,
                        size_t& hitRatio,
                        size_t& rmseMean,
                        const size_t numRecs)
{
  // Variable declarations.
  arma::Col<size_t> userVec(numItems), itemScore(numItems);
  arma::mat predictors;

  size_t norm, rmse;
  arma::Col<size_t> hits(testData.n_cols), rmses(testData.n_cols);

  for (size_t i = 0; i < testData.n_cols; i++)
  {
    // Considered user item rating test data.
    size_t u = testData(0, i);
    size_t gtItem = testData(1, i);
    size_t rt = testData(2, i);

    // Get negatives of items.
    arma::Col<size_t> itemVec(100);
    itemVec.rows(0, 98) = (arma::conv_to< arma::Col<size_t> >::from(
        negatives[u])).rows(0, 98);
    itemVec(99) = size_t(gtItem);

    // Form input for the network.
    userVec.fill(u);
    predictors = arma::conv_to< arma::mat >::from(
        arma::join_vert(userVec, itemVec));

    arma::mat results;
    network.Predict(predictors, results);

    // Find root mean squared error of predicted rating of considered item.
    for (size_t j = 0; j < itemVec.n_elem; j++)
    {
      itemScore[itemVec[j]] = results[j];
      if (gtItem == itemVec[j])
      {
        norm = (itemScore[itemVec[j]] * 4) + 1;
        rmse = ((rt - norm) ^ 2);
      }
    }
    size_t hr = 0;

    // Find if the item has been predicted in top k.
    for (size_t k = 0; k < numRecs; k++)
    {
      if (arma::as_scalar(arma::find(results.max() == results, 1)) == gtItem)
      {
        hr = 1;
      }
      itemScore(arma::as_scalar(
          arma::find(itemScore.max() == itemScore, 1))) = 0;
    }
    hits.insert_rows(i, hr);
    rmses.insert_rows(i, rmse);
  }

  // Find hit ratio and root mean squared error.
  hitRatio = arma::mean(hits);
  rmseMean = std::sqrt(arma::mean(rmses));
}

/**
 * Get recommendations for all users.
 */
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
void NCF::GetRecommendations(const size_t numRecs,
                             arma::Mat<size_t>& recommendations,
                             const arma::Col<size_t>& users)
{
  // Column vector of all items.
  arma::Col<size_t> itemVec = arma::linspace<arma::Col<size_t> >(0,
      numItems - 1, numItems);
  arma::Col<size_t> userVec(numItems);
  arma::mat predictors;

  // Predict rating for all user item combinations for given users.
  for (size_t i = 0; i < users.n_elem; i++)
  {
    arma::mat results;
    userVec.fill(users[i]);

    // Form input for the network.
    predictors = arma::conv_to< arma::mat >::from(
        arma::join_vert(userVec, itemVec));
    network.Predict(predictors, results);

    // Find top k recommendations.
    for (size_t k = 0; k < numRecs; k++)
    {
      recommendations(k, i) = (size_t) (arma::as_scalar(
          arma::find(results.max() == results, 1)));
      results(arma::as_scalar(arma::find(results.max() == results, 1))) = 0;
    }
  }
}

/**
 * Serialize the NCF model to the given archive.
 */
template<typename Archive>
void NCF::serialize(Archive& ar, const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(neg);
  ar & BOOST_SERIALIZATION_NVP(epochs);
  ar & BOOST_SERIALIZATION_NVP(embedSize);
  ar & BOOST_SERIALIZATION_NVP(numUsers);
  ar & BOOST_SERIALIZATION_NVP(numItems);
}

} // namespace cf
} // namespace mlpack

#endif
