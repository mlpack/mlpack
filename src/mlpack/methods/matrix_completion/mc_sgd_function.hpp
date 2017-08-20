/**
 * @file mc_sgd_function.hpp
 * @author Chenzhe Diao
 *
 * Optimization object function for matrix completion problem.
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_SGD_FUNCTION_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_SGD_FUNCTION_HPP

namespace mlpack {
namespace matrix_completion {

class MatrixCompletionSGDFunction {
 public:
  MatrixCompletionSGDFunction(const size_t m,
                              const size_t n,
                              const arma::umat& indices,
                              const arma::vec& values,
                              const size_t r) :
      m(m), n(n), indices(indices), values(values), r(r),
      omegaRow(arma::zeros<arma::uvec>(m)),
      omegaCol(arma::zeros<arma::uvec>(n))
  { CountOmega(); }
  
  size_t NumFunctions() {return indices.n_cols};

  double Evaluate(const arma::mat& coordinates, const size_t i)
  {
    size_t leftId = indices(0, i);
    size_t rightId = indices(1, i);
    
    double f1 = arma::dot(coordinates.row(leftId), coordinates.row(rightId + m));
    f1 = std::pow(f - values(i), 2);
    
    double f2 = std::pow(arma::norm(coordinates.row(leftId), "fro"), 2);
    f2 = mu * f2 / 2.0 / omegaRow(leftId);
    
    double f3 = std::pow(arma::norm(coordinates.row(rightId + m), "fro"), 2);
    f3 = mu * f3 / 2.0 / omegaCol(rightId);
    
    return f1 + f2 + f3;
  }
  
  void Gradient(const arma::mat& coordinates,
                const size_t i,
                arma::mat& gradient)
  {
    gradient.zeros();
    size_t leftId = indices(0, i);
    size_t rightId = indices(1, i);
    
    arma::mat rowi = coordinates.row(leftId);
    arma::mat colj = coordinates.row(rightId + m);
    double f = arma::dot(rowi, colj);
    
    arma::mat gradientVec = ( mu / omegaRow(leftId) ) * rowi;
    gradientVec += ( f - values(i) ) * colj;
    gradient.row(leftId) = gradientVec;
    
    gradientVec = (mu / omegaCol(rightId)) * colj;
    gradientVec += ( f - values(i) ) * rowi;
    gradient.row(rightId + m) = gradientVec;
  }
  
 private:
  //! Number of rows of the matrix.
  size_t m;
  //! Number of columns of the matrix.
  size_t n;
  //! Indices for sparse matrix.
  arma::umat& indices;
  //! Values for sparse matrix.
  arma::vec& values;
  //! Rank of recovered matrix.
  size_t r;
  
  double mu = 1.0;
  
  arma::uvec omegaRow;
  arma::uvec omegaCol;
  
  void CountOmega()
  {
    for (size_t i = 0; i<indices.n_cols; i++) {
      omegaRow(indices(0, i))++;
      omegaCol(indices(1, i))++;
    }
  }
};
} // namespace matrix_completion
} // namespace mlpack


#endif
