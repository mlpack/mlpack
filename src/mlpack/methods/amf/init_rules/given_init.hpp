/**
 * @file given_initialization.hpp
 * @author Ryan Curtin
 *
 * Initialization rule for alternating matrix factorization (AMF). This simple
 * initialization is performed by assigning a given matrix to W and H.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_INIT_RULES_GIVEN_INIT_HPP
#define MLPACK_METHODS_AMF_INIT_RULES_GIVEN_INIT_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace amf {

/**
 * This initialization rule for AMF simply fills the W and H matrices with the
 * matrices given to the constructor of this object.  Note that this object does
 * not use std::move() during the Initialize() method, so it can be reused for
 * multiple AMF objects, but will incur copies of the W and H matrices.
 */
class GivenInitialization
{
 public:
  // Empty constructor required for the InitializeRule template.
  GivenInitialization() { }

  // Initialize the GivenInitialization object with the given matrices.
  GivenInitialization(const arma::mat& w, const arma::mat& h) : w(w), h(h) { }

  // Initialize the GivenInitialization object, taking control of the given
  // matrices.
  GivenInitialization(const arma::mat&& w, const arma::mat&& h) :
    w(std::move(w)),
    h(std::move(h))
  { }

  /**
   * Fill W and H with random uniform noise.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param W W matrix, to be filled with random noise.
   * @param H H matrix, to be filled with random noise.
   */
  template<typename MatType>
  inline void Initialize(const MatType& /* V */,
                         const size_t /* r */,
                         arma::mat& W,
                         arma::mat& H)
  {
    // Initialize to the given matrices.
    W = w;
    H = h;
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(w, "w");
    ar & data::CreateNVP(h, "h");
  }

 private:
  //! The W matrix for initialization.
  arma::mat w;
  //! The H matrix for initialization.
  arma::mat h;
};

} // namespace amf
} // namespace mlpack

#endif
