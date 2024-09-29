/**
 * @file methods/amf/init_rules/no_init.hpp
 * @author Ryan Curtin
 *
 * Initialization rule for alternating matrix factorization (AMF). This simple
 * initialization leaves matrices the way they are, only checking the size.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_AMF_INIT_RULES_NO_INIT_HPP
#define MLPACK_METHODS_AMF_INIT_RULES_NO_INIT_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {

/**
 * This initialization rule for AMF does not initialize W and H, but instead
 * simply sets them to the right size.  An exception is thrown if W or H is not
 * the correct size.
 */
class NoInitialization
{
 public:
  // Nothing to do in the constructor.
  NoInitialization() { }

  /**
   * Check that W and H are the right sizes.
   *
   * @param V Input matrix.
   * @param r Rank of decomposition.
   * @param W W matrix, to be initialized to given matrix.
   * @param H H matrix, to be initialized to given matrix.
   */
  template<typename MatType, typename WHMatType>
  static inline void Initialize(const MatType& V,
                                const size_t r,
                                WHMatType& W,
                                WHMatType& H)
  {
    // Make sure the initial W, H matrices have correct size.
    if (W.n_rows != V.n_rows || W.n_cols != r)
    {
      std::ostringstream oss;
      oss << "NoInitialization::Initialize(): W has incorrect size; expected "
          << V.n_rows << " x " << r << ", got " << W.n_rows << " x " << W.n_cols
          << "!";
      throw std::invalid_argument(oss.str());
    }

    if (H.n_rows != r || H.n_cols != V.n_cols)
    {
      std::ostringstream oss;
      oss << "NoInitialization::Initialize(): H has incorrect size; expected "
          << r << " x " << V.n_cols << ", got " << H.n_rows << " x " << H.n_cols
          << "!";
      throw std::invalid_argument(oss.str());
    }
  }

  // Initialize either W or H.
  template<typename MatType, typename WHMatType>
  static inline void InitializeOne(const MatType& V,
                                   const size_t r,
                                   WHMatType& M,
                                   const bool whichMatrix = true)
  {
    if (whichMatrix)
    {
      if (M.n_rows != V.n_rows || M.n_cols != r)
      {
        std::ostringstream oss;
        oss << "NoInitialization::Initialize(): W has incorrect size; expected "
            << V.n_rows << " x " << r << ", got " << M.n_rows << " x "
            << M.n_cols << "!";
        throw std::invalid_argument(oss.str());
      }
    }
    else
    {
      if (M.n_rows != r || M.n_cols != V.n_cols)
      {
        std::ostringstream oss;
        oss << "NoInitialization::Initialize(): H has incorrect size; expected "
            << r << " x " << V.n_cols << ", got " << M.n_rows << " x "
            << M.n_cols << "!";
        throw std::invalid_argument(oss.str());
      }
    }
  }

  //! Serialize the object (in this case, there is nothing to serialize).
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */) { }
};

} // namespace mlpack

#endif
