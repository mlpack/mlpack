/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_em_main.cpp
 *
 * This program test drives the parametric estimation
 * of a Gaussian Mixture model using maximum likelihood.
 *
 * PARAMETERS TO BE INPUT:
 *
 * --data
 * This is the file that contains the data on which
 * the model is to be fit
 *
 * --mog_em/K
 * This is the number of gaussians we want to fit
 * on the data, defaults to '1'
 *
 * --output
 * This file will contain the parameters estimated,
 * defaults to 'ouotput.csv'
 *
 */
#include "mog_em.hpp"

PROGRAM_INFO("Mixture of Gaussians",
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using maximum likelihood.", "mog");

PARAM_STRING_REQ("data", "A file containing the data on which the model has to "
    "be fit.", "mog");
PARAM_STRING("output", "The file into which the output is to be written into.",
    "mog", "output.csv");

using namespace mlpack;
using namespace mlpack::gmm;

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  arma::mat data_points;
  data::Load(CLI::GetParam<std::string>("mog/data").c_str(), data_points, true);

  ////// MIXTURE OF GAUSSIANS USING EM //////
  MoGEM mog;

  CLI::GetParam<int>("mog/k") = 1;
  CLI::GetParam<int>("mog/d") = data_points.n_rows;

  ////// Timing the initialization of the mixture model //////
  Timers::StartTimer("mog/model_init");
  mog.Init(1, data_points.n_rows);
  Timers::StopTimer("mog/model_init");

  ////// Computing the parameters of the model using the EM algorithm //////
  std::vector<double> results;

  Timers::StartTimer("mog/EM");
  mog.ExpectationMaximization(data_points);
  Timers::StopTimer("mog/EM");

  mog.Display();
  mog.OutputResults(results);

  ////// OUTPUT RESULTS //////
  // We need a better solution for this.  So, currently, we do nothing.
  // XML is probably the right tool for the job.
}
