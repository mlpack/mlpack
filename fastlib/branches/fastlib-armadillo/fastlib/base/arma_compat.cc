/***
 * @file arma_compat.cc
 *
 * Compatibility layer for GenMatrix and arma::mat classes.
 *
 */

#include "arma_compat.h"

/**
 * Turn the supplied arma::mat into a FASTLIB Matrix.
 *
 * This wasn't written to be fast or efficient.
 */
void arma_compat::armaToMatrix(const arma::mat& mat, Matrix& mg) {
  // initialize our matrix correctly
  mg.Init(mat.n_rows, mat.n_cols);

  // copy elementwise.  this is slow as hell
  for(int r = 0; r < mat.n_rows; r++) {
    for(int c = 0; c < mat.n_cols; c++) {
      mg.set(r, c, mat(r, c));
    }
  }
}

/**
 * Turn the supplied FASTLIB Matrix into an arma::mat object.
 *
 * This wasn't written to be fast or efficient.
 */
void arma_compat::matrixToArma(const Matrix& gm, arma::mat& mat) {
  // initialize our matrix to the correct size
  mat.set_size(gm.n_rows(), gm.n_cols());

  // copy elementwise.  this is slow as hell
  for(int r = 0; r < gm.n_rows(); r++) {
    for(int c = 0; c < gm.n_cols(); c++) {
      mat(r, c) = gm.get(r, c);
    }
  }
}

/***
 * Write the selected column of the matrix into the uninitialized Vector.
 *
 * This wasn't written to be fast or efficient.
 */
void arma_compat::armaColVector(const arma::mat& mat, int col, Vector& v) {
  // initialize our vector to the correct size
  v.Init(mat.n_rows);

  // copy elementwise.  this is slow as hell
  for(int r = 0; r < mat.n_rows; r++)
    v[r] = mat(r, col);
}
