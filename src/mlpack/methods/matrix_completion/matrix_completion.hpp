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
                   const arma::umat& indices,
                   const arma::vec& values,
                   const size_t r);

  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values,
                   const arma::mat& initialPoint);

  MatrixCompletion(const size_t m,
                   const size_t n,
                   const arma::umat& indices,
                   const arma::vec& values);

  void Recover();

  const optimization::LRSDP& Sdp() const { return sdp; }
  optimization::LRSDP& Sdp() { return sdp; }

  const arma::mat& Recovered() const { return recovered; }
  arma::mat& Recovered() { return recovered; }

private:
  size_t m;
  size_t n;
  arma::umat indices;
  arma::mat values;

  optimization::LRSDP sdp;
  arma::mat recovered;

  void checkValues();
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
