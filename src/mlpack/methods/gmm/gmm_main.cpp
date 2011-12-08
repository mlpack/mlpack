/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file gmm_main.cpp
 *
 * This program trains a mixture of Gaussians on a given data matrix.
 */
#include "gmm.hpp"

PROGRAM_INFO("GMM",
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using the EM algorithm to find the maximum likelihood estimate.");

PARAM_STRING_REQ("data", "A file containing the data on which the model has to "
    "be fit.", "D");
PARAM_INT("gaussians", "g", "G", 1);

using namespace mlpack;
using namespace mlpack::gmm;

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  arma::mat data_points;
  data::Load(CLI::GetParam<std::string>("data").c_str(), data_points, true);

  ////// MIXTURE OF GAUSSIANS USING EM //////
  GMM gmm(CLI::GetParam<int>("gaussians"), data_points.n_rows);

  ////// Computing the parameters of the model using the EM algorithm //////
  Timers::StartTimer("em");
  gmm.Estimate(data_points);
  Timers::StopTimer("em");

  ////// OUTPUT RESULTS //////
  // We need a better solution for this.  So, currently, we do nothing.
  // XML is probably the right tool for the job.
}
