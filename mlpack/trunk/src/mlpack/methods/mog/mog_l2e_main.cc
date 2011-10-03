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

int main(int argc, char* argv[]) {
  IO::ParseCommandLine(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  arma::mat data_points;
  data::Load(IO::GetParam<std::string>("mog_l2e/data").c_str(), data_points);

  ////// MIXTURE OF GAUSSIANS USING L2 ESTIMATION //////
  size_t number_of_gaussians = IO::GetParam<int>("mog_l2e/k");
  IO::GetParam<int>("mog_l2e/d") = data_points.n_rows;
  size_t dimension = IO::GetParam<int>("mog_l2e/d");

  ////// RUNNING AN OPTIMIZER TO MINIMIZE THE L2 ERROR //////
  const char *opt_method = IO::GetParam<std::string>("opt/method").c_str();
  size_t param_dim = (number_of_gaussians * (dimension + 1) * (dimension + 2)
      / 2 - 1);
  IO::GetParam<int>("opt/param_space_dim") = param_dim;

  size_t optim_flag = (strcmp(opt_method, "NelderMead") == 0 ? 1 : 0);
  MoGL2E mog;

  if (optim_flag == 1) {
    ////// OPTIMIZER USING NELDER MEAD METHOD //////
    NelderMead opt;

    ////// Initializing the optimizer //////
    IO::StartTimer("opt/init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points);
    IO::StopTimer("opt/init_opt");

    ////// Getting starting points for the optimization //////
    arma::mat pts(param_dim, param_dim + 1);

    IO::StartTimer("opt/get_init_pts");
    MoGL2E::MultiplePointsGenerator(pts, data_points, number_of_gaussians);
    IO::StopTimer("opt/get_init_pts");

    ////// The optimization //////
    IO::StartTimer("opt/optimizing");
    opt.Eval(pts);
    IO::StopTimer("opt/optimizing");

    ////// Making model with the optimal parameters //////
    // This is a stupid way to do it and putting the 0s there ensures it will
    // fail.  Do it better!
    mog.MakeModel(0, 0, pts.col(0));
  } else {
    ////// OPTIMIZER USING QUASI NEWTON METHOD //////
    QuasiNewton opt;

    ////// Initializing the optimizer //////
    IO::StartTimer("opt/init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points);
    IO::StopTimer("opt/init_opt");

    ////// Getting starting point for the optimization //////
    arma::vec pt(param_dim);

    IO::StartTimer("opt/get_init_pt");
    MoGL2E::InitialPointGenerator(pt, data_points, number_of_gaussians);
    IO::StopTimer("opt/get_init_pt");

    ////// The optimization //////
    IO::StartTimer("opt/optimizing");
    opt.Eval(pt);
    IO::StopTimer("opt/optimizing");

    ////// Making model with optimal parameters //////
    // This is a stupid way to do it and putting the 0s there ensures it will
    // fail.  Do it better!
    mog.MakeModel(0, 0, pt);

  }

  long double error = mog.L2Error(data_points);
  IO::Info << "Minimum L2 error achieved: " << error << "." << std::endl;
  mog.Display();

  std::vector<double> results;
  mog.OutputResults(results);

  ////// OUTPUT RESULTS //////
  // We need a better way to do this (like XML).  For now we just do nothing.
}
