/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file mog_em_main.cc
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

#include "mog_em.h"

/*const fx_entry_doc mog_em_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   " A file containing the data on which the model"
   " has to be fit.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   " The file into which the output is to be written into.\n"},
  FX_ENTRY_DOC_DONE
};*/

PARAM_STRING_REQ("data", "A file containing the data on which the model has to be fit.", "mog");
PARAM_STRING("output", "The file into which the output is to be written into.", "mog", "output.csv");

/*const fx_submodule_doc mog_em_main_submodules[] = {
  {"mog_em", &mog_em_doc,
   " Responsible for intializing the model and"
   " computing the parameters.\n"},
  FX_SUBMODULE_DOC_DONE
};*/

/*const fx_module_doc mog_em_main_doc = {
  mog_em_main_entries, mog_em_main_submodules,
  " This program test drives the parametric estimation "
  "of a Gaussian mixture model using maximum likelihood.\n"
};*/




int main(int argc, char* argv[]) {
  
  IO::ParseCommandLine(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  
  const char *data_filename = IO::GetParam<std::string>("mog/data").c_str();

  Matrix data_points;
  data::Load(data_filename, &data_points);

  ////// MIXTURE OF GAUSSIANS USING EM //////

  MoGEM mog;

  IO::GetParam<int>("mog/K") = 1;
  IO::GetParam<int>("mog/D") = data_points.n_rows());

  ////// Timing the initialization of the mixture model //////
  IO::StartTimer("mog/model_init");
  mog.Init(mog_em_module);
  IO::StopTimer("mog/model_init");

  ////// Computing the parameters of the model using the EM algorithm //////
  ArrayList<double> results;

  IO::StartTimer("mog/EM");
  mog.ExpectationMaximization(data_points);
  IO::StopTimer("mog/EM");
  
  mog.Display();
  mog.OutputResults(&results);

  ////// OUTPUT RESULTS //////

  const char *output_filename = IO::GetParam<std::string>("mog/output").c_str();

  FILE *output_file = fopen(output_filename, "w");

  ot::Print(results, output_file);
  fclose(output_file);

  return 1;
}
