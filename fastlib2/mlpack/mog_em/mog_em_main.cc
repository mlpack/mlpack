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

#include "mog.h"

const fx_entry_doc mog_em_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
   " A file containing the data on which the model"
   " has to be fit.\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   " The file into which the output is to be written into.\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc mog_em_main_submodules[] = {
  {"mog_em", &mog_em_doc,
   " Responsible for intializing the model and"
   " computing the parameters.\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc mog_em_main_doc = {
  mog_em_main_entries, mog_em_main_submodules,
  " This program test drives the parametric estimation "
  "of a Gaussian mixture model using maximum likelihood.\n"
};

int main(int argc, char* argv[]) {

  fx_module *root = 
    fx_init(argc, argv, &mog_em_main_doc);

  ////// READING PARAMETERS AND LOADING DATA //////
  
  const char *data_filename = fx_param_str_req(root, "data");

  Matrix data_points;
  data::Load(data_filename, &data_points);

  ////// MIXTURE OF GAUSSIANS USING EM //////

  MoGEM mog;

  struct datanode* mog_em_module = 
    fx_submodule(root, "mog_em");
  fx_param_int(mog_em_module, "K", 1);
  fx_set_param_int(mog_em_module, "D", data_points.n_rows());

  ////// Timing the initialization of the mixture model //////
  fx_timer_start(mog_em_module, "model_init");
  mog.Init(mog_em_module);
  fx_timer_stop(mog_em_module, "model_init");

  ////// Computing the parameters of the model using the EM algorithm //////
  ArrayList<double> results;

  fx_timer_start(mog_em_module, "EM");
  mog.ExpectationMaximization(data_points);
  fx_timer_stop(mog_em_module, "EM");
  
  mog.Display();
  mog.OutputResults(&results);

  ////// OUTPUT RESULTS //////

  const char *output_filename = fx_param_str(NULL, "output", "output.csv");

  FILE *output_file = fopen(output_filename, "w");

  ot::Print(results, output_file);
  fclose(output_file);
  fx_done(root);

  return 1;
}
