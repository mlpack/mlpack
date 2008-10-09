/** @file nwrcde_main.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "nwrcde.h"

int main(int argc, char *argv[]) {
  
  // Initialize FastExec...
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "nwrcde_module" under the
  // root directory (NULL) for the NwrCde object to work inside.
  // Here, we initialize it with all parameters defined
  // "--nwrcde/...=...".
  struct datanode* nwrcde_module = fx_submodule(fx_root, "nwrcde");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");
  
  // The file containing the reference target values is a required
  // parameter.
  const char* reference_targets_file_name = fx_param_str_req
    (fx_root, "dtarget");

  // The query data file defaults to the references.
  const char* queries_file_name = fx_param_str(fx_root, "query",
					       references_file_name);

  // query and reference datasets and target training values.
  Matrix references;
  Matrix reference_targets;
  Matrix queries;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);  
  data::Load(queries_file_name, &queries);
  data::Load(reference_targets_file_name, &reference_targets);

  // Declare the computation object.
  NWRCde<GaussianKernel> algorithm;

  // Finalize FastExec and print output results.
  fx_done(fx_root);
  return 0;
}
