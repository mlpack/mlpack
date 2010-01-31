/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file lle_main.cc
 * 
 * This program test drives the process of obtaining 
 * the lower dimensional embedding of a given data
 * by performing nonlinear dimensionality reduction 
 * by locally linear embedding
 * 
 * PARAMETERS TO BE INPUT:
 * 
 * --data 
 * This is the file that contains the data on which 
 * the nonlinear dimensionality reduction is 
 * to be performed.
 *
 * --lle/knn
 * This is the number of nearest neighbors to be 
 * considered to obtain the locally linear 
 * embedding.
 *
 * --lle/d
 * This is the dimension to which the higher 
 * dimensional data is to be embedded upon, defaults 
 * to 2.
 *
 * --output
 * This file will contain the parameters estimated,
 * defaults to 'output.arff'.
 *
 */

#include "lle.h"

int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  ////// LOADING DATA //////
  
  const char *data_filename = fx_param_str_req(NULL, "data");

  Matrix dataset;
  data::Load(data_filename, &dataset);

  ////// LOCAL LINEAR EMBEDDING //////


  // The Init function of LLE pulls out all the parameters from the 
  // 'lle_module' which takes in inputs from the command line. But the 
  // user can hardcode the parameters in the main function by 
  // changing the default values of the module parameters to the desired 
  // values and not inputting anything from the command line

  datanode *lle_module = fx_submodule(NULL, "lle", "lle");
  index_t lower_dimension = fx_param_int(lle_module, "d", 2);
  index_t dimension = fx_param_int(lle_module, "D", dataset.n_rows());
  fx_param_int(lle_module, "N", dataset.n_cols());  

  DEBUG_ASSERT_MSG(lower_dimension <= dimension, 
		   "The maximum embedding dimensionality > data dimension\n");
  LLE lle;

  // The number of nearest neighbours to be considered for obtaining 
  // the local embedding
  fx_param_int(lle_module, "knn", 5); 

  ////// Initializing the LLE object ///////

  fx_timer_start(lle_module, "object_init");
  lle.Init(dataset, lle_module);
  fx_timer_stop(lle_module, "object_init");

  ////// Computing the lower dimensional embedding //////

  fx_timer_start(lle_module, "embedding");
  lle.Compute();
  fx_timer_stop(lle_module, "embedding");    

  ////// Obtaining results //////


  // changes made as per review to make output more readable

  Matrix lower_dimensional_embedding, actual_output_mat;
  lle.OutputResults(&lower_dimensional_embedding);
  la::TransposeInit(lower_dimensional_embedding, &actual_output_mat);

  ////// OUTPUT RESULTS //////

  const char *output_filename = fx_param_str(NULL, "output", "output.arff");

  FILE *output_file = fopen(output_filename, "w");

  // changes made as per review to make output more readable

  actual_output_mat.PrintDebug("Lower Dimensional Embedding",
			       output_file);
  NOTIFY("Lower dimensional embeddings of %"LI"d points in  %"LI"d dimensions obtained\n",
	 lower_dimensional_embedding.n_cols(),
	 lower_dimensional_embedding.n_rows());

  fclose(output_file);

  fx_done();

  return 1;
}
