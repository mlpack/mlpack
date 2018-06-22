/**
 * @file nmf_method.hpp
 * @author Haritha Nair
 *
 * Implementation of the exact svd method for use in Collaborative Filtering.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_NMF_METHOD_HPP
#define MLPACK_METHODS_CF_DECOMPOSITION_POLICIES_NMF_METHOD_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/update_rules/nmf_als.hpp>
#include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
#include <mlpack/methods/amf/termination_policies/simple_residue_termination.hpp>

namespace mlpack {
namespace cf {

/**
 * Implementation of the NMF policy to act as a wrapper when accessing
 * NMF from within CFType.
 */
class NMFPolicy
{
 public:
  /**
   * Apply Collaborative Filtering to the provided dataset using NMF method.
   *
   * @param data Data matrix: dense matrix (coordinate lists) 
   *    or sparse matrix(cleaned).
   * @param cleanedData item user table in form of sparse matrix.
   * @param rank Rank parameter for matrix factorization.
   * @param w First matrix formed after decomposition.
   * @param h Second matrix formed after decomposition.
   * @param maxIterations Maximum number of iterations.
   * @param minResidue Residue required to terminate.
   * @param mit Whether to terminate only when maxIterations is reached.
   */
  template<typename MatType>
  void Apply(const MatType& /* data */,
             const arma::sp_mat& cleanedData,
             const size_t rank,
             arma::mat& w,
             arma::mat& h,
             const size_t maxIterations,
             const double minResidue,
             const bool mit)
  {
    if (mit)
    {
      amf::MaxIterationTermination iter(maxIterations);

      // Do singular value decomposition using the NMF algorithm.
      amf::AMF<amf::MaxIterationTermination, amf::RandomInitialization,
          amf::NMFALSUpdate> nmf(iter);
      nmf.Apply(cleanedData, rank, w, h);
    }
    else
    {
      amf::SimpleResidueTermination srt(minResidue, maxIterations);

      // Do singular value decomposition using the NMF algorithm.
      amf::NMFALSFactorizer nmf(srt);
      nmf.Apply(cleanedData, rank, w, h);
    }
  }
};

} // namespace cf
} // namespace mlpack

#endif
