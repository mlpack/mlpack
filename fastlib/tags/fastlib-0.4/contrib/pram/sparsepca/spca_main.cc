/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file spca_main.cc
 * 
 * This program test drives the process of obtaining 
 * Sparse loadings for the Principal Components
 * of a given data.
 * 
 * PARAMETERS TO BE INPUT:
 * 
 * --data 
 * This is the file that contains the data on which 
 * the principal component analysis is done
 *
 * --spca/K
 * This is the number of components with sparse
 * loading requested; defaults to '1'
 *
 * --output
 * This file will contain the parameters estimated,
 * defaults to 'output.arff'
 *
 */

#include "sparsepca.h"

int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  ////// LOADING DATA //////
  
  const char *data_filename = fx_param_str_req(NULL, "data");

  Matrix dataset;
  data::Load(data_filename, &dataset);

  ////// SPARSE PRINCIPAL COMPONENT ANALYSIS //////

  datanode *spca_module = fx_submodule(NULL, "spca", "spca");
  index_t number_of_components = fx_param_int(spca_module, "K", 1);
  index_t dimension = fx_param_int(spca_module, "D", dataset.n_rows());
  index_t n = fx_param_int(spca_module, "N", dataset.n_cols());  

  DEBUG_ASSERT_MSG(number_of_components <= dimension, 
		   "The number of principal components > data dimension\n");
  SparsePCA spca;
  double lambda_2 = 0.0;
  ArrayList<double> lambda_1;

  lambda_1.Init(number_of_components);
  for (index_t i = 0; i < number_of_components; i++) {
    lambda_1[i] = 0.01;
  }

  ////// Initializing the SparsePCA object ///////

  fx_timer_start(spca_module, "object_init");
  spca.Init(dataset, lambda_2, lambda_1, spca_module);
  fx_timer_stop(spca_module, "object_init");

  ////// Computing the sparse principal components //////

  fx_timer_start(spca_module, "sparsification");
  if (dimension / n > 100) { // for the case where dim >> n
    NOTIFY("Micro Array Analysis\n");
    spca.SparsifyMicroArray();
  }
  else { // the general SPC algorithm
    NOTIFY("General Analysis\n");
    spca.Sparsify();
  }
  fx_timer_stop(spca_module, "sparsification");    

  ////// Obtaining results //////

  Matrix sparse_loadings;
  double percent_variance_explained = spca.OutputResults(&sparse_loadings);

  ////// OUTPUT RESULTS //////

  const char *output_filename = fx_param_str(NULL, "output", "output.arff");

  FILE *output_file = fopen(output_filename, "w");

  fprintf(output_file, "@ Percent Variance Explained : %f\n", 
	  percent_variance_explained);
  fprintf(output_file, "@ Sparse Loadings\n");

  index_t rows = sparse_loadings.n_rows(), cols = sparse_loadings.n_cols();
  for (index_t i = 0; i < rows; i++) {
    for (index_t j = 0; j < cols; j++) {
      fprintf(output_file, "%f,", sparse_loadings.get(i, j));
    }
    fprintf(output_file, "\b\n");
  }

  fclose(output_file);

  fx_done();

  return 1;
}
