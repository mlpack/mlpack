/**
 * @file lars_main.cpp
 * @author Nishant Mehta
 *
 * Executable for LARS.
 *
 * This file is part of MLPACK 1.0.2.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <mlpack/core.hpp>

#include "lars.hpp"

PROGRAM_INFO("LARS", "An implementation of LARS: Least Angle Regression "
    "(Stagewise/laSso).  This is a stage-wise homotopy-based algorithm for "
    "L1-regularized linear regression (LASSO) and L1+L2-regularized linear "
    "regression (Elastic Net).\n"
    "\n"
    "Let X be a matrix where each row is a point and each column is a "
    "dimension, and let y be a vector of targets.\n"
    "\n"
    "The Elastic Net problem is to solve\n\n"
    "  min_beta 0.5 || X * beta - y ||_2^2 + lambda_1 ||beta||_1 +\n"
    "      0.5 lambda_2 ||beta||_2^2\n\n"
    "If lambda_1 > 0 and lambda_2 = 0, the problem is the LASSO.\n"
    "If lambda_1 > 0 and lambda_2 > 0, the problem is the Elastic Net.\n"
    "If lambda_1 = 0 and lambda_2 > 0, the problem is Ridge Regression.\n"
    "If lambda_1 = 0 and lambda_2 = 0, the problem is unregularized linear "
    "regression.\n"
    "\n"
    "For efficiency reasons, it is not recommended to use this algorithm with "
    "lambda_1 = 0.\n");

PARAM_STRING_REQ("input_file", "File containing covariates (X).",
    "i");
PARAM_STRING_REQ("responses_file", "File containing y "
    "(responses/observations).", "r");

PARAM_STRING("output_file", "File to save beta (linear estimator) to.", "o",
    "output.csv");

PARAM_DOUBLE("lambda1", "Regularization parameter for l1-norm penalty.", "l",
    0);
PARAM_DOUBLE("lambda2", "Regularization parameter for l2-norm penalty.", "L",
    0);
PARAM_FLAG("use_cholesky", "Use Cholesky decomposition during computation "
    "rather than explicitly computing the full Gram matrix.", "c");

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;

int main(int argc, char* argv[])
{
  // Handle parameters,
  CLI::ParseCommandLine(argc, argv);

  double lambda1 = CLI::GetParam<double>("lambda1");
  double lambda2 = CLI::GetParam<double>("lambda2");
  bool useCholesky = CLI::HasParam("use_cholesky");

  // Load covariates.  We can avoid LARS transposing our data by choosing to not
  // transpose this data.
  const string matXFilename = CLI::GetParam<string>("input_file");
  mat matX;
  data::Load(matXFilename.c_str(), matX, true, false);

  // Load responses.  The responses should be a one-dimensional vector, and it
  // seems more likely that these will be stored with one response per line (one
  // per row).  So we should not transpose upon loading.
  const string yFilename = CLI::GetParam<string>("responses_file");
  mat matY; // Will be a vector.
  data::Load(yFilename.c_str(), matY, true, false);

  // Make sure y is oriented the right way.
  if (matY.n_rows == 1)
    matY = trans(matY);
  if (matY.n_cols > 1)
    Log::Fatal << "Only one column or row allowed in responses file!" << endl;

  if (matY.n_elem != matX.n_rows)
    Log::Fatal << "Number of responses must be equal to number of rows of X!"
        << endl;

  // Do LARS.
  LARS lars(useCholesky, lambda1, lambda2);
  vec beta;
  lars.Regress(matX, matY.unsafe_col(0), beta, false /* do not transpose */);

  const string betaFilename = CLI::GetParam<string>("output_file");
  beta.save(betaFilename, raw_ascii);
}
