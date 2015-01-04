/**
 * @file matrix_completion.hpp
 * @author Stephen Tu
 *
 * A thin wrapper around nuclear norm minimization to solve
 * low rank matrix completion problems.
 */
#ifndef __MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP
#define __MLPACK_METHODS_MATRIX_COMPLETION_MATRIX_COMPLETION_HPP

#include <mlpack/core/optimizers/lrsdp/lrsdp.hpp>

namespace mlpack {
namespace matrix_completion {

class MatrixCompletion
{
public:
  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::mat& entries,
                   const size_t r);

  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::mat& entries,
                   const arma::mat& initialPoint);

  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::mat& entries);

  void Recover();

  const arma::mat& Recovered() const { return recovered; }
  arma::mat& Recovered() { return recovered; }

private:
  size_t m;
  size_t n;
  arma::mat entries;

  optimization::LRSDP sdp;
  arma::mat recovered;

  void initSdp();

  static size_t
  DefaultRank(const size_t m,
              const size_t n,
              const size_t p);

  static arma::mat
  CreateInitialPoint(const size_t m,
                     const size_t n,
                     const size_t r);

};

} // namespace matrix_completion
} // namespace mlpack

// Include implementation.
#include "matrix_completion_impl.hpp"

#endif
