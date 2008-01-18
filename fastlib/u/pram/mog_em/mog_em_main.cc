#include "mog.h"


int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////
  
  const char *data_filename = fx_param_str_req(NULL, "data");

  Matrix data_points;
  data::Load(data_filename, &data_points);

  ////// MIXTURE OF GAUSSIANS USING EM //////

  MoG mog;

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
  mog.Display();

  fx_done();

  return 0;
}

/*
 Omega : [ 0.090901 0.809896 0.099203 ]
 Mu : 
[-1.742582 1.109790 3.150051 ;-0.056386 0.453788 -0.277957 ;-0.252493 0.077252 -0.753365 ]
Sigma : 
[1.830775 1.278463 1.293301 ;1.278463 0.981999 0.938991 ;1.293301 0.938991 1.026536 ]
[1.878899 1.302212 1.305705 ;1.302212 1.747098 0.554765 ;1.305705 0.554765 1.197182 ]
[30.584888 1.965165 -2.471019 ;1.965165 35.255788 9.399539 ;-2.471019 9.399539 35.458573 ]
*/
