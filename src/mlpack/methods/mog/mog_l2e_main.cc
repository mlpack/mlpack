/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_l2e_main.cc
 *
 * This program test drives the L2 estimation
 * of a Gaussian Mixture model.
 *
 * PARAMETERS TO BE INPUT:
 *
 * --data
 * This is the file that contains the data on which
 * the model is to be fit
 *
 * --mog_l2e/K
 * This is the number of gaussians we want to fit
 * on the data, defaults to '1'
 *
 * --output
 * This file will contain the parameters estimated,
 * defaults to 'output.csv'
 *
 */
#include "mog_l2e.h"
#include "optimizers.h"

PROGRAM_INFO("Mixture of Gaussians",
    "This program takes a parametric estimate of a Gaussian mixture model (GMM)"
    " using the L2 loss function.", "mog_l2e");

PARAM_STRING_REQ("data", "A file containing the data on which the model has to "
    "be fit.", "mog_l2e");
PARAM_STRING("output", "The file into which the output is to be written into",
    "mog_l2e", "output.csv");

using namespace mlpack;
using namespace mlpack::gmm;

int main(int argc, char* argv[]) {
  CLI::ParseCommandLine(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  arma::mat data_points;
  data_points.load(CLI::GetParam<std::string>("mog_l2e/data").c_str(),
      arma::auto_detect, false, true);

  ////// MIXTURE OF GAUSSIANS USING L2 ESTIMATCLIN //////
  size_t number_of_gaussians = CLI::GetParam<int>("mog_l2e/k");
  CLI::GetParam<int>("mog_l2e/d") = data_points.n_rows;
  size_t dimension = CLI::GetParam<int>("mog_l2e/d");

  ////// RUNNING AN OPTIMIZER TO MINIMIZE THE L2 ERROR //////
  const char *opt_method = CLI::GetParam<std::string>("opt/method").c_str();
  size_t param_dim = (number_of_gaussians * (dimension + 1) * (dimension + 2)
      / 2 - 1);
  CLI::GetParam<int>("opt/param_space_dim") = param_dim;

  size_t optim_flag = (strcmp(opt_method, "NelderMead") == 0 ? 1 : 0);
  MoGL2E mog;

  if (optim_flag == 1) {
    ////// OPTIMIZER USING NELDER MEAD METHOD //////
    NelderMead opt;

    ////// Initializing the optimizer //////
    CLI::StartTimer("opt/init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points);
    CLI::StopTimer("opt/init_opt");

    ////// Getting starting points for the optimization //////
    arma::mat pts(param_dim, param_dim + 1);

    CLI::StartTimer("opt/get_init_pts");
    MoGL2E::MultiplePointsGenerator(pts, data_points, number_of_gaussians);
    CLI::StopTimer("opt/get_init_pts");

    ////// The optimization //////
    CLI::StartTimer("opt/optimizing");
    opt.Eval(pts);
    CLI::StopTimer("opt/optimizing");

    ////// Making model with the optimal parameters //////
    // This is a stupid way to do it and putting the 0s there ensures it will
    // fail.  Do it better!
    mog.MakeModel(0, 0, pts.col(0));
  } else {
    ////// OPTIMIZER USING QUASI NEWTON METHOD //////
    QuasiNewton opt;

    ////// Initializing the optimizer //////
    CLI::StartTimer("opt/init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points);
    CLI::StopTimer("opt/init_opt");

    ////// Getting starting point for the optimization //////
    arma::vec pt(param_dim);

    CLI::StartTimer("opt/get_init_pt");
    MoGL2E::InitialPointGenerator(pt, data_points, number_of_gaussians);
    CLI::StopTimer("opt/get_init_pt");

    ////// The optimization //////
    CLI::StartTimer("opt/optimizing");
    opt.Eval(pt);
    CLI::StopTimer("opt/optimizing");

    ////// Making model with optimal parameters //////
    // This is a stupid way to do it and putting the 0s there ensures it will
    // fail.  Do it better!
    mog.MakeModel(0, 0, pt);

  }

  long double error = mog.L2Error(data_points);
  Log::Info << "Minimum L2 error achieved: " << error << "." << std::endl;
  mog.Display();

  std::vector<double> results;
  mog.OutputResults(results);

  ////// OUTPUT RESULTS //////
  // We need a better way to do this (like XML).  For now we just do nothing.
}
