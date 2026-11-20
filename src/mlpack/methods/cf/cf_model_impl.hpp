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
#ifndef MLPACK_METHODS_CF_CF_MODEL_IMPL_HPP
#define MLPACK_METHODS_CF_CF_MODEL_IMPL_HPP

#include "cf_model.hpp"

namespace mlpack {

inline CFModel::CFModel() :
    decompositionType(NMF),
    normalizationType(NO_NORMALIZATION),
    cf(NULL)
{
  // Nothing else to do.
}

inline CFModel::CFModel(const CFModel& other) :
    decompositionType(other.decompositionType),
    normalizationType(other.normalizationType),
    cf(other.cf->Clone())
{
  // Nothing else to do.
}

inline CFModel::CFModel(CFModel&& other) :
    decompositionType(other.decompositionType),
    normalizationType(other.normalizationType),
    cf(std::move(other.cf))
{
  // Reset properties of the other one.
  other.decompositionType = NMF;
  other.normalizationType = NO_NORMALIZATION;
}

inline CFModel& CFModel::operator=(const CFModel& other)
{
  if (this != &other)
  {
    decompositionType = other.decompositionType;
    normalizationType = other.normalizationType;
    cf = other.cf->Clone();
  }

  return *this;
}

inline CFModel& CFModel::operator=(CFModel&& other)
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

inline CFModel::~CFModel()
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

    case CFModel::ITEM_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, ItemMeanNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);

    case CFModel::USER_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, UserMeanNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);

    case CFModel::OVERALL_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, OverallMeanNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);

    case CFModel::Z_SCORE_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, ZScoreNormalization>(data,
          decomposition, numUsersForSimilarity, rank, maxIterations, minResidue,
          mit);
  }

  // This shouldn't ever happen.
  return NULL;
}

template<typename NeighborSearchPolicy, typename CFType>
void PredictHelper(CFType& cf,
                   const InterpolationTypes interpolationType,
                   const arma::Mat<size_t>& combinations,
                   arma::vec& predictions)
{
  switch (interpolationType)
  {
    case AVERAGE_INTERPOLATION:
      cf.template Predict<NeighborSearchPolicy,
                          AverageInterpolation>(combinations, predictions);
      break;

    case REGRESSION_INTERPOLATION:
      cf.template Predict<NeighborSearchPolicy,
                          RegressionInterpolation>(combinations, predictions);
      break;

    case SIMILARITY_INTERPOLATION:
      cf.template Predict<NeighborSearchPolicy,
                          SimilarityInterpolation>(combinations, predictions);
      break;
  }
}

//! Make predictions.
template<typename DecompositionPolicy, typename NormalizationPolicy>
void CFWrapper<DecompositionPolicy, NormalizationPolicy>::Predict(
    const NeighborSearchTypes nsType,
    const InterpolationTypes interpolationType,
    const arma::Mat<size_t>& combinations,
    arma::vec& predictions)
{
  switch (nsType)
  {
    case COSINE_SEARCH:
      PredictHelper<CosineSearch>(cf, interpolationType, combinations,
          predictions);
      break;

    case EUCLIDEAN_SEARCH:
      PredictHelper<EuclideanSearch>(cf, interpolationType, combinations,
          predictions);
      break;

    case PEARSON_SEARCH:
      PredictHelper<PearsonSearch>(cf, interpolationType, combinations,
          predictions);
      break;
  }
}

template<typename NeighborSearchPolicy, typename CFType>
void GetRecommendationsHelper(
    CFType& cf,
    const InterpolationTypes interpolationType,
    const size_t numRecs,
    arma::Mat<size_t>& recommendations,
    const arma::Col<size_t>& users)
{
  switch (interpolationType)
  {
    case AVERAGE_INTERPOLATION:
      cf.template GetRecommendations<NeighborSearchPolicy,
                                     AverageInterpolation>(
          numRecs, recommendations, users);
      break;

    case REGRESSION_INTERPOLATION:
      cf.template GetRecommendations<NeighborSearchPolicy,
                                     RegressionInterpolation>(
          numRecs, recommendations, users);
      break;

    case SIMILARITY_INTERPOLATION:
      cf.template GetRecommendations<NeighborSearchPolicy,
                                     SimilarityInterpolation>(
          numRecs, recommendations, users);
      break;
  }
}

//! Compute recommendations for queried users.
template<typename DecompositionPolicy, typename NormalizationPolicy>
void CFWrapper<DecompositionPolicy, NormalizationPolicy>::GetRecommendations(
    const NeighborSearchTypes nsType,
    const InterpolationTypes interpolationType,
    const size_t numRecs,
    arma::Mat<size_t>& recommendations,
    const arma::Col<size_t>& users)
{
  switch (nsType)
  {
    case COSINE_SEARCH:
      GetRecommendationsHelper<CosineSearch>(cf, interpolationType, numRecs,
          recommendations, users);
      break;

    case EUCLIDEAN_SEARCH:
      GetRecommendationsHelper<EuclideanSearch>(cf, interpolationType, numRecs,
          recommendations, users);
      break;

    case PEARSON_SEARCH:
      GetRecommendationsHelper<PearsonSearch>(cf, interpolationType, numRecs,
          recommendations, users);
      break;
  }
}

template<typename NeighborSearchPolicy, typename CFType>
void GetRecommendationsHelper(
    CFType& cf,
    const InterpolationTypes interpolationType,
    const size_t numRecs,
    arma::Mat<size_t>& recommendations)
{
  switch (interpolationType)
  {
    case AVERAGE_INTERPOLATION:
      cf.template GetRecommendations<NeighborSearchPolicy,
                                     AverageInterpolation>(
          numRecs, recommendations);
      break;

    case REGRESSION_INTERPOLATION:
      cf.template GetRecommendations<NeighborSearchPolicy,
                                     RegressionInterpolation>(
          numRecs, recommendations);
      break;

    case SIMILARITY_INTERPOLATION:
      cf.template GetRecommendations<NeighborSearchPolicy,
                                     SimilarityInterpolation>(
          numRecs, recommendations);
      break;
  }
}

//! Compute recommendations for all users.
template<typename DecompositionPolicy, typename NormalizationPolicy>
void CFWrapper<DecompositionPolicy, NormalizationPolicy>::GetRecommendations(
    const NeighborSearchTypes nsType,
    const InterpolationTypes interpolationType,
    const size_t numRecs,
    arma::Mat<size_t>& recommendations)
{
  switch (nsType)
  {
    case COSINE_SEARCH:
      GetRecommendationsHelper<CosineSearch>(cf, interpolationType, numRecs,
          recommendations);
      break;

    case EUCLIDEAN_SEARCH:
      GetRecommendationsHelper<EuclideanSearch>(cf, interpolationType, numRecs,
          recommendations);
      break;

    case PEARSON_SEARCH:
      GetRecommendationsHelper<PearsonSearch>(cf, interpolationType, numRecs,
          recommendations);
      break;
  }
}

template<typename DecompositionPolicy>
CFWrapperBase* InitializeModelHelper(
    CFModel::NormalizationTypes normalizationType)
{
  switch (normalizationType)
  {
    case CFModel::NO_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, NoNormalization>();

    case CFModel::ITEM_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, ItemMeanNormalization>();

    case CFModel::USER_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, UserMeanNormalization>();

    case CFModel::OVERALL_MEAN_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, OverallMeanNormalization>();

    case CFModel::Z_SCORE_NORMALIZATION:
      return new CFWrapper<DecompositionPolicy, ZScoreNormalization>();
  }

  // This shouldn't ever happen.
  return NULL;
}

inline CFWrapperBase* InitializeModel(
    CFModel::DecompositionTypes decompositionType,
    CFModel::NormalizationTypes normalizationType)
{
  switch (decompositionType)
  {
    case CFModel::NMF:
      return InitializeModelHelper<NMFPolicy>(normalizationType);

    case CFModel::BATCH_SVD:
      return InitializeModelHelper<BatchSVDPolicy>(normalizationType);

    case CFModel::RANDOMIZED_SVD:
      return InitializeModelHelper<RandomizedSVDPolicy>(normalizationType);

    case CFModel::REG_SVD:
      return InitializeModelHelper<RegSVDPolicy>(normalizationType);

    case CFModel::SVD_COMPLETE:
      return InitializeModelHelper<SVDCompletePolicy>(normalizationType);

    case CFModel::SVD_INCOMPLETE:
      return InitializeModelHelper<SVDIncompletePolicy>(normalizationType);

    case CFModel::BIAS_SVD:
      return InitializeModelHelper<BiasSVDPolicy>(normalizationType);

    case CFModel::SVD_PLUS_PLUS:
      return InitializeModelHelper<SVDPlusPlusPolicy>(normalizationType);

    case CFModel::QUIC_SVD:
      return InitializeModelHelper<QUIC_SVDPolicy>(normalizationType);

    case CFModel::BLOCK_KRYLOV_SVD:
      return InitializeModelHelper<BlockKrylovSVDPolicy>(normalizationType);
  }

  // This shouldn't ever happen.
  return NULL;
};

template<typename DecompositionPolicy, typename Archive>
void SerializeHelper(Archive& ar,
                     CFWrapperBase* cf,
                     CFModel::NormalizationTypes normalizationType)
{
  switch (normalizationType)
  {
    case CFModel::NO_NORMALIZATION:
      {
        CFWrapper<DecompositionPolicy, NoNormalization>& typedModel =
            dynamic_cast<CFWrapper<DecompositionPolicy,
                                   NoNormalization>&>(*cf);
        ar(CEREAL_NVP(typedModel));
        break;
      }

    case CFModel::ITEM_MEAN_NORMALIZATION:
      {
        CFWrapper<DecompositionPolicy, ItemMeanNormalization>& typedModel =
            dynamic_cast<CFWrapper<DecompositionPolicy,
                                   ItemMeanNormalization>&>(*cf);
        ar(CEREAL_NVP(typedModel));
        break;
      }

    case CFModel::USER_MEAN_NORMALIZATION:
      {
        CFWrapper<DecompositionPolicy, UserMeanNormalization>& typedModel =
            dynamic_cast<CFWrapper<DecompositionPolicy,
                                   UserMeanNormalization>&>(*cf);
        ar(CEREAL_NVP(typedModel));
        break;
      }

    case CFModel::OVERALL_MEAN_NORMALIZATION:
      {
        CFWrapper<DecompositionPolicy, OverallMeanNormalization>& typedModel =
            dynamic_cast<CFWrapper<DecompositionPolicy,
                                   OverallMeanNormalization>&>(*cf);
        ar(CEREAL_NVP(typedModel));
        break;
      }

    case CFModel::Z_SCORE_NORMALIZATION:
      {
        CFWrapper<DecompositionPolicy, ZScoreNormalization>& typedModel =
            dynamic_cast<CFWrapper<DecompositionPolicy,
                                   ZScoreNormalization>&>(*cf);
        ar(CEREAL_NVP(typedModel));
        break;
      }
  }
}

inline void CFModel::Train(
    const arma::mat& data,
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

    case QUIC_SVD:
      cf = TrainHelper(QUIC_SVDPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;

    case BLOCK_KRYLOV_SVD:
      cf = TrainHelper(BlockKrylovSVDPolicy(), normalizationType, data,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
      break;
  }
}

//! Make predictions.
inline void CFModel::Predict(
    const NeighborSearchTypes nsType,
    const InterpolationTypes interpolationType,
    const arma::Mat<size_t>& combinations,
    arma::vec& predictions)
{
  cf->Predict(nsType, interpolationType, combinations, predictions);
}

//! Compute recommendations for queried users.
inline void CFModel::GetRecommendations(
    const NeighborSearchTypes nsType,
    const InterpolationTypes interpolationType,
    const size_t numRecs,
    arma::Mat<size_t>& recommendations,
    const arma::Col<size_t>& users)
{
  cf->GetRecommendations(nsType, interpolationType, numRecs, recommendations,
      users);
}

//! Compute recommendations for all users.
inline void CFModel::GetRecommendations(
    const NeighborSearchTypes nsType,
    const InterpolationTypes interpolationType,
    const size_t numRecs,
    arma::Mat<size_t>& recommendations)
{
  cf->GetRecommendations(nsType, interpolationType, numRecs, recommendations);
}

template<typename Archive>
void CFModel::serialize(Archive& ar, const uint32_t /* version */)
{
  ar(CEREAL_NVP(decompositionType));
  ar(CEREAL_NVP(normalizationType));

  // This should never happen, but just in case, be clean with memory.
  if (cereal::is_loading<Archive>())
  {
    delete cf;
    cf = InitializeModel(decompositionType, normalizationType);
  }

  // Avoid polymorphic serialization by determining the type directly.
  switch (decompositionType)
  {
    case NMF:
      SerializeHelper<NMFPolicy>(ar, cf, normalizationType);
      break;

    case BATCH_SVD:
      SerializeHelper<BatchSVDPolicy>(ar, cf, normalizationType);
      break;

    case RANDOMIZED_SVD:
      SerializeHelper<RandomizedSVDPolicy>(ar, cf, normalizationType);
      break;

    case REG_SVD:
      SerializeHelper<RegSVDPolicy>(ar, cf, normalizationType);
      break;

    case SVD_COMPLETE:
      SerializeHelper<SVDCompletePolicy>(ar, cf, normalizationType);
      break;

    case SVD_INCOMPLETE:
      SerializeHelper<SVDIncompletePolicy>(ar, cf, normalizationType);
      break;

    case BIAS_SVD:
      SerializeHelper<BiasSVDPolicy>(ar, cf, normalizationType);
      break;

    case SVD_PLUS_PLUS:
      SerializeHelper<SVDPlusPlusPolicy>(ar, cf, normalizationType);
      break;

    case QUIC_SVD:
      SerializeHelper<QUIC_SVDPolicy>(ar, cf, normalizationType);
      break;

    case BLOCK_KRYLOV_SVD:
      SerializeHelper<BlockKrylovSVDPolicy>(ar, cf, normalizationType);
      break;
  }
}

} // namespace mlpack

#endif
