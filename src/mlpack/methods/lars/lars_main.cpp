/**
 * @file lars_main.cpp
 * @author Nishant Mehta
 *
 * Executable for LARS
 */

#include <mlpack/core.hpp>
#include <armadillo>
#include "lars.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::lars;

PROGRAM_INFO("LARS", "An implementation of LARS: Least Angle Regression (Stagewise/laSso)");

PARAM_STRING_REQ("X", "Covariates filename (observations of input random "
		 "variables)", "");
PARAM_STRING_REQ("y", "Targets filename (observations of output random "
		 "variable", "");
PARAM_STRING_REQ("beta", "Solution filename (linear estimator)", "");

PARAM_DOUBLE("lambda1", "Regularization parameter for l1-norm penalty", "", 0);
PARAM_DOUBLE("lambda2", "Regularization parameter for l2-norm penalty", "", 0);
PARAM_FLAG("use_cholesky", "Use Cholesky decomposition during computation "
	   "rather than explicitly computing full Gram matrix", "");


int main(int argc, char* argv[])
{
  
  // Handle parameters
  CLI::ParseCommandLine(argc, argv);
  
  double lambda1 = CLI::GetParam<double>("lambda1");
  double lambda2 = CLI::GetParam<double>("lambda2");
  bool useCholesky = CLI::GetParam<bool>("use_cholesky");

  // load covariates
  const std::string matXFilename = CLI::GetParam<std::string>("X");
  mat matX;
  matX.load(matXFilename, raw_ascii);
  
  // load targets
  const std::string yFilename = CLI::GetParam<std::string>("y");
  vec y;
  y.load(yFilename, raw_ascii);
  
  // do LARS
  LARS lars(matX, y, useCholesky, lambda1, lambda2);
  lars.DoLARS();
  
  // get and save solution
  vec beta;
  lars.Solution(beta);
  
  const std::string betaFilename = CLI::GetParam<std::string>("beta");
  beta.save(betaFilename, raw_ascii);
}
