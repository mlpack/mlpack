/**
 * @file methods/cf/cf_model.hpp
 * @author Wenhao Huang
 * @author Khizir Siddiqui
 *
 * A serializable CF model, used by the main program.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_CF_MODEL_HPP
#define MLPACK_METHODS_CF_CF_MODEL_HPP

#include <mlpack/core.hpp>
#include "cf.hpp"

namespace mlpack {

/**
 * NeighborSearchTypes contains the set of NeighborSearchPolicy classes that are
 * usable by CFModel at prediction time.
 */
enum NeighborSearchTypes
{
  COSINE_SEARCH,
  EUCLIDEAN_SEARCH,
  PEARSON_SEARCH
};

/**
 * InterpolationTypes contains the set of InterpolationPolicy classes that are
 * usable by CFModel at prediction time.
 */
enum InterpolationTypes
{
  AVERAGE_INTERPOLATION,
  REGRESSION_INTERPOLATION,
  SIMILARITY_INTERPOLATION
};

/**
 * The CFWrapperBase class provides a unified interface that can be used by the
 * CFModel class to interact with all different CF types at runtime.  All CF
 * wrapper types inherit from this base class.
 */
class CFWrapperBase
{
 public:
  //! Create the object.  The base class has nothing to hold.
  CFWrapperBase() { }

  //! Make a copy of the object.
  virtual CFWrapperBase* Clone() const = 0;

  //! Delete the object.
  virtual ~CFWrapperBase() { }

  //! Compute predictions for users.
  virtual void Predict(const NeighborSearchTypes nsType,
                       const InterpolationTypes interpolationType,
                       const arma::Mat<size_t>& combinations,
                       arma::vec& predictions) = 0;

  //! Compute recommendations for all users.
  virtual void GetRecommendations(
      const NeighborSearchTypes nsType,
      const InterpolationTypes interpolationType,
      const size_t numRecs,
      arma::Mat<size_t>& recommendations) = 0;

  //! Compute recommendations.
  virtual void GetRecommendations(
      const NeighborSearchTypes nsType,
      const InterpolationTypes interpolationType,
      const size_t numRecs,
      arma::Mat<size_t>& recommendations,
      const arma::Col<size_t>& users) = 0;
};

/**
 * The CFWrapper class wraps the functionality of all CF types.  If special
 * handling is needed for a future CF type, this class can be extended.
 */
template<typename DecompositionPolicy, typename NormalizationPolicy>
class CFWrapper : public CFWrapperBase
{
 protected:
  using CFModelType = CFType<DecompositionPolicy, NormalizationPolicy>;

 public:
  //! Create the CFWrapper object, using default parameters to initialize the
  //! held CF object.
  CFWrapper() { }

  //! Create the CFWrapper object, initializing the held CF object.
  CFWrapper(const arma::mat& data,
            const DecompositionPolicy& decomposition,
            const size_t numUsersForSimilarity,
            const size_t rank,
            const size_t maxIterations,
            const size_t minResidue,
            const bool mit) :
      cf(data,
         decomposition,
         numUsersForSimilarity,
         rank,
         maxIterations,
         minResidue,
         mit)
  {
    // Nothing else to do.
  }

  //! Clone the CFWrapper object.  This handles polymorphism correctly.
  virtual CFWrapper* Clone() const { return new CFWrapper(*this); }

  //! Destroy the CFWrapper object.
  virtual ~CFWrapper() { }

  //! Get the CFType object.
  CFModelType& CF() { return cf; }

  //! Compute predictions for users.
  virtual void Predict(const NeighborSearchTypes nsType,
                       const InterpolationTypes interpolationType,
                       const arma::Mat<size_t>& combinations,
                       arma::vec& predictions);

  //! Compute recommendations for all users.
  virtual void GetRecommendations(
      const NeighborSearchTypes nsType,
      const InterpolationTypes interpolationType,
      const size_t numRecs,
      arma::Mat<size_t>& recommendations);

  //! Compute recommendations.
  virtual void GetRecommendations(
      const NeighborSearchTypes nsType,
      const InterpolationTypes interpolationType,
      const size_t numRecs,
      arma::Mat<size_t>& recommendations,
      const arma::Col<size_t>& users);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(cf));
  }

 protected:
  //! This is the CF object that we are wrapping.
  CFModelType cf;
};

/**
 * The model to save to disk.
 */
class CFModel
{
 public:
  enum DecompositionTypes
  {
    NMF,
    BATCH_SVD,
    RANDOMIZED_SVD,
    REG_SVD,
    SVD_COMPLETE,
    SVD_INCOMPLETE,
    BIAS_SVD,
    SVD_PLUS_PLUS,
    QUIC_SVD,
    BLOCK_KRYLOV_SVD
  };

  enum NormalizationTypes
  {
    NO_NORMALIZATION,
    ITEM_MEAN_NORMALIZATION,
    USER_MEAN_NORMALIZATION,
    OVERALL_MEAN_NORMALIZATION,
    Z_SCORE_NORMALIZATION
  };

 private:
  //! The current decomposition policy type.
  DecompositionTypes decompositionType;
  //! The current normalization policy type.
  NormalizationTypes normalizationType;

  /**
   * cf holds an instance of the CFType class for the current
   * decompositionPolicy and normalizationType. It is initialized every time
   * Train() is executed.
   */
  CFWrapperBase* cf;

 public:
  //! Create an empty CF model.
  CFModel();

  //! Create a CF model by copying the given model.
  CFModel(const CFModel& other);

  //! Create a CF model by taking ownership of the data of the other model.
  CFModel(CFModel&& other);

  //! Make this CF model a copy of the other model.
  CFModel& operator=(const CFModel& other);

  //! Make this CF model take ownership of the data of the other model.
  CFModel& operator=(CFModel&& other);

  //! Clean up memory.
  ~CFModel();

  //! Get the CFWrapperBase object.  (Be careful!)
  CFWrapperBase* CF() const { return cf; }

  //! Get the decomposition type.
  const DecompositionTypes& DecompositionType() const
  {
    return decompositionType;
  }
  //! Set the decomposition type.
  DecompositionTypes& DecompositionType()
  {
    return decompositionType;
  }

  //! Get the normalization type.
  const NormalizationTypes& NormalizationType() const
  {
    return normalizationType;
  }
  //! Set the normalization type.
  NormalizationTypes& NormalizationType()
  {
    return normalizationType;
  }

  //! Train the model.
  void Train(const arma::mat& data,
             const size_t numUsersForSimilarity,
             const size_t rank,
             const size_t maxIterations,
             const double minResidue,
             const bool mit);

  //! Make predictions.
  void Predict(const NeighborSearchTypes nsType,
               const InterpolationTypes interpolationType,
               const arma::Mat<size_t>& combinations,
               arma::vec& predictions);

  //! Compute recommendations for query users.
  void GetRecommendations(const NeighborSearchTypes nsType,
                          const InterpolationTypes interpolationType,
                          const size_t numRecs,
                          arma::Mat<size_t>& recommendations,
                          const arma::Col<size_t>& users);

  //! Compute recommendations for all users.
  void GetRecommendations(const NeighborSearchTypes nsType,
                          const InterpolationTypes interpolationType,
                          const size_t numRecs,
                          arma::Mat<size_t>& recommendations);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);
};

} // namespace mlpack

// Include implementation.
#include "cf_model_impl.hpp"

#endif
