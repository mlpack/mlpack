/**
 * @file cf_model.cpp
 * @author Wenhao Huang
 *
 * A serializable CF model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "cf.hpp"
#include "cf_model.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::cf;

//! Create an empty CF model.
CFModel::CFModel() :
    decompositionPolicy(0),
    nmfCF(NULL),
    batchSVDCF(NULL),
    randSVDCF(NULL),
    regSVDCF(NULL),
    completeSVDCF(NULL),
    incompleteSVDCF(NULL)
{
  // Nothing to do.
}

CFModel::~CFModel()
{
  delete nmfCF;
  delete batchSVDCF;
  delete randSVDCF;
  delete regSVDCF;
  delete completeSVDCF;
  delete incompleteSVDCF;
}

//! Make predictions.
void CFModel::Predict(const arma::Mat<size_t>& combinations,
                      arma::vec& predictions)
{
  switch (decompositionPolicy)
  {
    case DecompositionPolicies::NMF:
      nmfCF->Predict(combinations, predictions);
      break;
    case DecompositionPolicies::BATCH_SVD:
      batchSVDCF->Predict(combinations, predictions);
      break;
    case DecompositionPolicies::RANDOMIZED_SVD:
      randSVDCF->Predict(combinations, predictions);
      break;
    case DecompositionPolicies::REGULARIZED_SVD:
      regSVDCF->Predict(combinations, predictions);
      break;
    case DecompositionPolicies::SVD_COMPLETE:
      completeSVDCF->Predict(combinations, predictions);
      break;
    case DecompositionPolicies::SVD_INCOMPLETE:
      incompleteSVDCF->Predict(combinations, predictions);
      break;
  }
}

//! Compute recommendations for query users.
void CFModel::GetRecommendations(const size_t numRecs,
                                 arma::Mat<size_t>& recommendations,
                                 const arma::Col<size_t>& users)
{
  switch (decompositionPolicy)
  {
    case DecompositionPolicies::NMF:
      nmfCF->GetRecommendations(numRecs, recommendations, users);
      break;
    case DecompositionPolicies::BATCH_SVD:
      batchSVDCF->GetRecommendations(numRecs, recommendations, users);
      break;
    case DecompositionPolicies::RANDOMIZED_SVD:
      randSVDCF->GetRecommendations(numRecs, recommendations, users);
      break;
    case DecompositionPolicies::REGULARIZED_SVD:
      regSVDCF->GetRecommendations(numRecs, recommendations, users);
      break;
    case DecompositionPolicies::SVD_COMPLETE:
      completeSVDCF->GetRecommendations(numRecs, recommendations, users);
      break;
    case DecompositionPolicies::SVD_INCOMPLETE:
      incompleteSVDCF->GetRecommendations(numRecs, recommendations, users);
      break;
  }
}

//! Compute recommendations for all users.
void CFModel::GetRecommendations(const size_t numRecs,
                                 arma::Mat<size_t>& recommendations)
{
  switch (decompositionPolicy)
  {
    case DecompositionPolicies::NMF:
      nmfCF->GetRecommendations(numRecs, recommendations);
      break;
    case DecompositionPolicies::BATCH_SVD:
      batchSVDCF->GetRecommendations(numRecs, recommendations);
      break;
    case DecompositionPolicies::RANDOMIZED_SVD:
      randSVDCF->GetRecommendations(numRecs, recommendations);
      break;
    case DecompositionPolicies::REGULARIZED_SVD:
      regSVDCF->GetRecommendations(numRecs, recommendations);
      break;
    case DecompositionPolicies::SVD_COMPLETE:
      completeSVDCF->GetRecommendations(numRecs, recommendations);
      break;
    case DecompositionPolicies::SVD_INCOMPLETE:
      incompleteSVDCF->GetRecommendations(numRecs, recommendations);
      break;
  }
}
