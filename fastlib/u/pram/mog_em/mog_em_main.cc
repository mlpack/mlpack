/**
 * @author pram
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
 * --number_of_gaussians
 * This is the number of gaussians we want to fit
 * on the data, defaults to '1'
 *
 * --output_filename
 * This file will contain the parameters estimated,
 * defaults to 'ouotput.csv'
 *
 */

#include "mog.h"


int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  
  const char *data_filename = fx_param_str_req(NULL, "data");

  Matrix data_points;
  data::Load(data_filename, &data_points);

  ////// MIXTURE OF GAUSSIANS USING EM //////

  MoGEM mog;

  struct datanode* mog_em_module = fx_submodule(NULL, "mog_em", "mog_em_module");
  const int number_of_gaussians = fx_param_int(NULL, "number_of_gaussians", 1);
  const int dimensions = data_points.n_rows();

  ////// Timing the initialization of the mixture model //////
  fx_timer_start(mog_em_module, "model_initializing");

  mog.Init(number_of_gaussians, dimensions);

  fx_timer_stop(mog_em_module, "model_initializing");

  ////// Computing the parameters of the model using the EM algorithm //////
  ArrayList<double> results;

  fx_timer_start(mog_em_module, "optimization_via_EM");

  mog.ExpectationMaximization(data_points, &results);

  fx_timer_stop(mog_em_module, "optimization_via_EM");

  ////// OUTPUT RESULTS //////

  const char *output_filename = fx_param_str(NULL, "output_filename", "output.csv");

  FILE *output_file = fopen(output_filename, "w");

  ot::Print(results, output_file);
  fclose(output_file);
  fx_done();

  return 1;
}
