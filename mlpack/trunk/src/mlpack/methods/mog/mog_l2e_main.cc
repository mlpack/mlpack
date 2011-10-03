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

/*const fx_entry_doc mog_l2e_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   " A file containing the data on which the model"
   " has to be fit.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   " The file into which the output is to be written into.\n"},
  FX_ENTRY_DOC_DONE
};*/

PARAM_STRING_REQ("data", "A file containing the data on\
 which the model has to fit", "mog_l2e");
PARAM_STRING("output", "The file into which the output \
is to be written into", "mog_l2e", "output.csv");

/*const fx_submodule_doc mog_l2e_main_submodules[] = {
  {"mog_l2e", &mog_l2e_doc,
   " Responsible for intializing the model and"
   " computing the parameters.\n"},
   {"opt", &opt_doc,
    " Responsible for minimizing the L2 loss function"
    " and obtaining the parameter values.\n"},
  FX_SUBMODULE_DOC_DONE
};*/

PARAM_MODULE("mog_l2e", "Responsible for initializing the\
 model and computing the parameters.");
PARAM_MODULE("opt", "Responsible for minimizing the L2 loss\
 function and obtaining the parameter values");

/*const fx_module_doc mog_l2e_main_doc = {
  mog_l2e_main_entries, mog_l2e_main_submodules,
  " This program test drives the parametric estimation "
  "of a Gaussian mixture model using L2 loss function.\n"
};*/

PROGRAM_INFO("MOG", "This program test drives the parametric estimation\
 of a Gaussian mixture model using L2 loss function.");

using namespace mlpack;

int main(int argc, char* argv[]) {

  IO::ParseCommandLine(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////

  const char *data_filename = IO::GetParam<std::string>("mog/data").c_str();

  Matrix data_points;
  data::Load(data_filename, &data_points);

  ////// MIXTURE OF GAUSSIANS USING L2 ESTIMATION //////

  size_t number_of_gaussians = IO::GetParam<int>("mog_l2e/K");
  IO::GetParam<int>("mog_l2e/D") = data_points.n_rows());
  size_t dimension = IO::GetParam<int>("mog_l2e/D");

  ////// RUNNING AN OPTIMIZER TO MINIMIZE THE L2 ERROR //////

  const char *opt_method = IO::GetParam<std::string>("opt/method");
  size_t param_dim = (number_of_gaussians*(dimension+1)*(dimension+2)/2 - 1);
  IO::GetParam<int>("opt/param_space_dim") = param_dim;

  size_t optim_flag = (strcmp(opt_method, "NelderMead") == 0 ? 1 : 0);
  MoGL2E mog;

  if (optim_flag == 1) {

    ////// OPTIMIZER USING NELDER MEAD METHOD //////

    NelderMead opt;

    ////// Initializing the optimizer //////
    IO::StartTimer("opt/init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points, opt_module);
    IO::StopTimer("opt/init_opt");

    ////// Getting starting points for the optimization //////
    double **pts;
    pts = (double**)malloc((param_dim+1)*sizeof(double*));
    for(size_t i = 0; i < param_dim+1; i++) {
      pts[i] = (double*)malloc(param_dim*sizeof(double));
    }

    IO::StartTimer("opt/get_init_pts");
    MoGL2E::MultiplePointsGenerator(pts, param_dim+1,
				    data_points, number_of_gaussians);
    IO::StopTimer("opt/get_init_pts");

    ////// The optimization //////

    IO::StartTimer("opt/optimizing");
    opt.Eval(pts);
    IO::StopTimer("opt/optimizing");

    ////// Making model with the optimal parameters //////
    mog.MakeModel(mog_l2e_module, pts[0]);

  }
  else {

    ////// OPTIMIZER USING QUASI NEWTON METHOD //////

    QuasiNewton opt;

    ////// Initializing the optimizer //////
    IO::StartTimer("opt/init_opt");
    opt.Init(MoGL2E::L2ErrorForOpt, data_points, opt_module);
    IO::StopTimer("opt/init_opt");

    ////// Getting starting point for the optimization //////
    double *pt;
    pt = (double*)malloc(param_dim*sizeof(double));

    IO::StartTimer("opt/get_init_pt");
    MoGL2E::InitialPointGenerator(pt, data_points, number_of_gaussians);
    IO::StopTimer("opt/get_init_pt");

    ////// The optimization //////

    IO::StartTimer("opt/optimizing");
    opt.Eval(pt);
    IO::StopTimer("opt/optimizing");

    ////// Making model with optimal parameters //////
    mog.MakeModel(mog_l2e_module, pt);

  }

  long double error = mog.L2Error(data_points);
  NOTIFY("Minimum L2 error achieved: %Lf", error);
  mog.Display();

  ArrayList<double> results;
  mog.OutputResults(&results);


  ////// OUTPUT RESULTS //////

  const char *output_filename = IO::GetParam<std::string>("mog_l2e/output");

  FILE *output_file = fopen(output_filename, "w");

  ot::Print(results, output_file);
  fclose(output_file);

  return 1;
}
