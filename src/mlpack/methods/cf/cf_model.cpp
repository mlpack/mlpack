/**
 * @file methods/cf/cf_model_impl.hpp
 * @author Wenhao Huang
 *
 * A serializable CF model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "cf_model.hpp"

namespace mlpack {
namespace cf {

CFModel::CFModel() :
    decompositionType(NMF),
    normalizationType(NO_NORMALIZATION),
    cf(NULL)
{
  // Nothing else to do.
}

CFModel::CFModel(const CFModel& other) :
    decompositionType(other.decompositionType),
    normalizationType(other.normalizationType),
    cf(other.cf->Clone())
{
  // Nothing else to do.
}

CFModel::CFModel(CFModel&& other) :
    decompositionType(other.decompositionType),
    normalizationType(other.normalizationType),
    cf(std::move(other.cf))
{
  // Reset properties of the other one.
  other.decompositionType = NMF;
  other.normalizationType = NO_NORMALIZATION;
}

CFModel& CFModel::operator=(const CFModel& other)
{
  if (this != &other)
  {
    decompositionType = other.decompositionType;
    normalizationType = other.normalizationType;
    cf = other.cf->Clone();
  }

  return *this;
}

CFModel& CFModel::operator=(CFModel&& other)
{
  if (this != &other)
  {
    decompositionType = other.decompositionType;
    normalizationType = other.normalizationType;
    cf = std::move(other.cf);

    // Reset the other object.
    other.decompositionType = NMF;
    other.normalizationType = NO_NORMALIZATION;
  }

  return *this;
}

CFModel::~CFModel()
{
  delete cf;
}

template<typename DecompositionPolicy>
CFWrapperBase* TrainHelper(const DecompositionPolicy& decomposition,
                           const CFModel::NormalizationTypes normalizationType,
                           const arma::mat& data,
                           const size_t numUsersForSimilarity,
                           const size_t rank,
                           const size_t maxIterations,
                           const double minResidue,
                           const bool mit)
{
  switch (normalizationType)
  {
    case CFModel::NO_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, NoNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);
      break;

    case CFModel::ITEM_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, ItemMeanNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);
      break;

    case CFModel::USER_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, UserMeanNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);
      break;

    case CFModel::OVERALL_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, OverallMeanNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);
      break;

    case CFModel::Z_SCORE_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, ZScoreNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);
      break;
  }

  // This shouldn't ever happen.
  return NULL;
}

void CFModel::Train(const arma::mat& data,
                    const size_t numUsersForSimilarity,
                    const size_t rank,
                    const size_t maxIterations,
                    const double minResidue,
                    const bool mit)
{
  // Delete the current CFType object, if there is one.
  delete cf;

  switch (decompositionType)
  {
    case NMF:
      cf = TrainHelper(NMFPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case BATCH_SVD:
      cf = TrainHelper(BatchSVDPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case RANDOMIZED_SVD:
      cf = TrainHelper(RandomizedSVDPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case REG_SVD:
      cf = TrainHelper(RegSVDPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case SVD_COMPLETE:
      cf = TrainHelper(SVDCompletePolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case SVD_INCOMPLETE:
      cf = TrainHelper(SVDIncompletePolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case BIAS_SVD:
      cf = TrainHelper(BiasSVDPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case SVD_PLUS_PLUS:
      cf = TrainHelper(SVDPlusPlusPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;
  }
}

//! Make predictions.
void CFModel::Predict(const NeighborSearchTypes nsType,
                      const InterpolationTypes interpolationType,
                      const arma::Mat<size_t>& combinations,
                      arma::vec& predictions)
{
  cf->Predict(nsType, interpolationType, combinations, predictions);
}

//! Compute recommendations for queried users.
void CFModel::GetRecommendations(const NeighborSearchTypes nsType,
                                 const InterpolationTypes interpolationType,
                                 const size_t numRecs,
                                 arma::Mat<size_t>& recommendations,
                                 const arma::Col<size_t>& users)
{
  cf->GetRecommendations(nsType, interpolationType, numRecs, recommendations,
      users);
}

//! Compute recommendations for all users.
void CFModel::GetRecommendations(const NeighborSearchTypes nsType,
                                 const InterpolationTypes interpolationType,
                                 const size_t numRecs,
                                 arma::Mat<size_t>& recommendations)
{
  cf->GetRecommendations(nsType, interpolationType, numRecs, recommendations);
}

} // namespace cf
} // namespace mlpack
