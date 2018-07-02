/**
 * @file cf_model.hpp
 * @author Wenhao Huang
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


#include <mlpack/methods/cf/decomposition_policies/batch_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/regularized_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svd_complete_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svd_incomplete_method.hpp>

namespace mlpack {
namespace cf {

/**
 * The model to save to disk.
 */
class CFModel
{
 public:
  enum DecompositionPolicies
  {
    NMF,
    BATCH_SVD,
    RANDOMIZED_SVD,
    REGULARIZED_SVD,
    SVD_COMPLETE,
    SVD_INCOMPLETE
  };

 private:
  //! The type of decomposition policy.
  size_t decompositionPolicy;
  //! Non-NULL if using NMFPolicy.
  CFType<NMFPolicy>* nmfCF;
  //! Non-NULL if using BatchSVDPolicy.
  CFType<BatchSVDPolicy>* batchSVDCF;
  //! Non-NULL if using RandomizedSVDPolicy.
  CFType<RandomizedSVDPolicy>* randSVDCF;
  //! Non-NULL if using RegSVDPolicy.
  CFType<RegSVDPolicy>* regSVDCF;
  //! Non-NULL if using SVDCompletePolicy.
  CFType<SVDCompletePolicy>* completeSVDCF;
  //! Non-NULL if using SVDIncompletePolicy.
  CFType<SVDIncompletePolicy>* incompleteSVDCF;

 public:
  //! Create an empty CF model.
  CFModel();

  //! Clean up memory.
  ~CFModel();

  //! Get the decomposition policy.
  size_t DecompositionPolicy() const { return decompositionPolicy; }
  //! Modify the decomposition policy.
  size_t& DecompositionPolicy() { return decompositionPolicy; }
  
  //! Get the pointer to CFType<> object.
  template<typename DecompositionPolicy>
  const CFType<DecompositionPolicy>* CFPtr() const
  {
    void* pointer = NULL;
    switch (decompositionPolicy)
    {
      case DecompositionPolicies::NMF:
        pointer = (void*) nmfCF;
        break;
      case DecompositionPolicies::BATCH_SVD:
        pointer = (void*) batchSVDCF;
        break;
      case DecompositionPolicies::RANDOMIZED_SVD:
        pointer = (void*) randSVDCF;
        break;
      case DecompositionPolicies::REGULARIZED_SVD:
        pointer = (void*) regSVDCF;
        break;
      case DecompositionPolicies::SVD_COMPLETE:
        pointer = (void*) completeSVDCF;
        break;
      case DecompositionPolicies::SVD_INCOMPLETE:
        pointer = (void*) incompleteSVDCF;
        break;
    }
    return (CFType<DecompositionPolicy>*) pointer;
  }

  //! Train the model.
  template <typename MatType>
  void Train(const MatType& data,
             const size_t numUsersForSimilarity,
             const size_t rank,
             const size_t maxIterations,
             const double minResidue,
             const bool mit)
  {
    if (decompositionPolicy == DecompositionPolicies::NMF)
    {
      NMFPolicy decomposition;
      nmfCF = new CFType<NMFPolicy>(data, decomposition,
            numUsersForSimilarity, rank, maxIterations, minResidue, mit);
    }
    else if (decompositionPolicy == DecompositionPolicies::BATCH_SVD)
    {
      BatchSVDPolicy decomposition;
      batchSVDCF = new CFType<BatchSVDPolicy>(data, decomposition,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
    }
    else if (decompositionPolicy == DecompositionPolicies::RANDOMIZED_SVD)
    {
      RandomizedSVDPolicy decomposition;
      randSVDCF = new CFType<RandomizedSVDPolicy>(data, decomposition,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
    }
    else if (decompositionPolicy == DecompositionPolicies::REGULARIZED_SVD)
    {  
      RegSVDPolicy decomposition;
      regSVDCF = new CFType<RegSVDPolicy>(data, decomposition,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
    }
    else if (decompositionPolicy == DecompositionPolicies::SVD_COMPLETE)
    {
      SVDCompletePolicy decomposition;
      completeSVDCF = new CFType<SVDCompletePolicy>(data, decomposition,
          numUsersForSimilarity, rank, maxIterations, minResidue, mit);
    }
    else if (decompositionPolicy == DecompositionPolicies::SVD_INCOMPLETE)
    {
        SVDIncompletePolicy decomposition;
        incompleteSVDCF = new CFType<SVDIncompletePolicy>(data, decomposition,
            numUsersForSimilarity, rank, maxIterations, minResidue, mit);
    }
  }

  //! Make predictions.
  void Predict(const arma::Mat<size_t>& combinations,
               arma::vec& predictions);

  //! Compute recommendations for query users.
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations,
                          const arma::Col<size_t>& users);
  
  //! Compute recommendations for all users.
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    if (Archive::is_loading::value)
    {
      if (nmfCF)
        delete nmfCF;
      if (batchSVDCF)
        delete batchSVDCF;
      if (randSVDCF)
        delete randSVDCF;
      if (regSVDCF)
        delete regSVDCF;
      if (completeSVDCF)
        delete completeSVDCF;
      if (incompleteSVDCF)
        delete incompleteSVDCF;

      nmfCF = NULL;
      batchSVDCF = NULL;
      randSVDCF = NULL;
      regSVDCF = NULL;
      completeSVDCF = NULL;
      incompleteSVDCF = NULL;
    }

    ar & BOOST_SERIALIZATION_NVP(decompositionPolicy);
    switch (decompositionPolicy)
    {
      case DecompositionPolicies::NMF:
        ar & BOOST_SERIALIZATION_NVP(nmfCF);
        break;
      case DecompositionPolicies::BATCH_SVD:
        ar & BOOST_SERIALIZATION_NVP(batchSVDCF);
        break;
      case DecompositionPolicies::RANDOMIZED_SVD:
        ar & BOOST_SERIALIZATION_NVP(randSVDCF);
        break;
      case DecompositionPolicies::REGULARIZED_SVD:
        ar & BOOST_SERIALIZATION_NVP(batchSVDCF);
        break;
      case DecompositionPolicies::SVD_COMPLETE:
        ar & BOOST_SERIALIZATION_NVP(completeSVDCF);
        break;
      case DecompositionPolicies::SVD_INCOMPLETE:
        ar & BOOST_SERIALIZATION_NVP(incompleteSVDCF);
        break;
    }
  }
};

} // namespace cf
} // namespace mlpack

#endif