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
#include <boost/variant.hpp>
#include "cf.hpp"

#include <mlpack/methods/cf/decomposition_policies/batch_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/randomized_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/regularized_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svd_complete_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svd_incomplete_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/bias_svd_method.hpp>
#include <mlpack/methods/cf/decomposition_policies/svdplusplus_method.hpp>

namespace mlpack {
namespace cf {

/**
 * DeleteVisitor deletes the CFType<> object which is pointed to by the
 * variable cf in class CFModel.
 */
class DeleteVisitor : public boost::static_visitor<void>
{
 public:
  //! Delete CFType object.
  template<typename DecompositionPolicy>
  void operator()(CFType<DecompositionPolicy>* c) const;
};

/**
 * GetValueVisitor returns the pointer which points to the CFType object.
 */
class GetValueVisitor : public boost::static_visitor<void*>
{
 public:
  //! Return stored pointer as void* type.
  template<typename DecompositionPolicy>
  void* operator()(CFType<DecompositionPolicy>* c) const;
};

/**
 * PredictVisitor uses the CFType object to make predictions on the given
 * combinations of users and items.
 */
template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
class PredictVisitor : public boost::static_visitor<void>
{
 private:
  //! User/item combinations to predict.
  const arma::Mat<size_t>& combinations;
  //! Predicted ratings for each user/item combination.
  arma::vec& predictions;

 public:
  //! Predict ratings for each user-item combination.
  template<typename DecompositionPolicy>
  void operator()(CFType<DecompositionPolicy>* c) const;

  //! Visitor constructor.
  PredictVisitor(const arma::Mat<size_t>& combinations,
                 arma::vec& predictions);
};

/**
 * RecommendationVisitor uses the CFType object to get recommendations for the
 * given users.
 */
template <typename NeighborSearchPolicy,
          typename InterpolationPolicy>
class RecommendationVisitor : public boost::static_visitor<void>
{
 private:
  //! Number of Recommendations.
  const size_t numRecs;
  //! Recommendations matrix to save recommendations.
  arma::Mat<size_t>& recommendations;
  //! Users for which recommendations are to be generated.
  const arma::Col<size_t>& users;
  //! Whether users are given.
  const bool usersGiven;

 public:
  //! Visitor constructor.
  RecommendationVisitor(const size_t numRecs,
                        arma::Mat<size_t>& recommendations,
                        const arma::Col<size_t>& users,
                        const bool usersGiven);

  //! Generates the given number of recommendations.
  template<typename DecompositionPolicy>
  void operator()(CFType<DecompositionPolicy>* c) const;
};

/**
 * The model to save to disk.
 */
class CFModel
{
 private:
  /**
   * cf holds an instance of the CFType class for the current
   * decompositionPolicy. It is initialized every time Train() is executed.
   * We access to the contained value through the visitor classes defined above.
   */
  boost::variant<CFType<NMFPolicy>*,
                 CFType<BatchSVDPolicy>*,
                 CFType<RandomizedSVDPolicy>*,
                 CFType<RegSVDPolicy>*,
                 CFType<SVDCompletePolicy>*,
                 CFType<SVDIncompletePolicy>*,
                 CFType<BiasSVDPolicy>*,
                 CFType<SVDPlusPlusPolicy>*> cf;

 public:
  //! Create an empty CF model.
  CFModel() { }

  //! Clean up memory.
  ~CFModel();

  //! Get the pointer to CFType<> object.
  template<typename DecompositionPolicy>
  const CFType<DecompositionPolicy>* CFPtr() const;

  //! Train the model.
  template<typename DecompositionPolicy,
           typename MatType>
  void Train(const MatType& data,
             const size_t numUsersForSimilarity,
             const size_t rank,
             const size_t maxIterations,
             const double minResidue,
             const bool mit);

  //! Make predictions.
  template <typename NeighborSearchPolicy,
            typename InterpolationPolicy>
  void Predict(const arma::Mat<size_t>& combinations,
               arma::vec& predictions);

  //! Compute recommendations for query users.
  template<typename NeighborSearchPolicy,
           typename InterpolationPolicy>
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations,
                          const arma::Col<size_t>& users);

  //! Compute recommendations for all users.
  template<typename NeighborSearchPolicy,
           typename InterpolationPolicy>
  void GetRecommendations(const size_t numRecs,
                          arma::Mat<size_t>& recommendations);

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */);
};

} // namespace cf
} // namespace mlpack

// Include implementation.
#include "cf_model_impl.hpp"

#endif
