// The file should include it's header
#include "allnn.h"


int main(int argc, char* argv[]) {
 
  // This line starts the FASTexec system for reading in parameters.
  // Always call it at the beginning of main to use FASTexec.
  // It's arguments are the same as main's.
  fx_init(argc, argv);
  
  
  // The filename where the queries are stored
  const char* queries_file_name = fx_param_str_req(NULL, "q");
  
  // Read in the data from the command line
  Matrix queries;
  data::Load(queries_file_name, &queries);
  
  // The same as above
  const char* references_file_name = fx_param_str_req(NULL, "r");
  Matrix references;
  data::Load(references_file_name, &references);
  
  // Make an AllNN object
  AllNN allnn;
  
  // We need a submodule to hold our allnn computation.
  struct datanode* allnn_module = fx_submodule(NULL, "allnn", "allnn_module");
  
  // We'll time the tree-building, which will be stored in the allnn module
  fx_timer_start(allnn_module, "tree_building");
  
  // Initializing the AllNN object with our parameters.
  allnn.Init(queries, references, allnn_module);
  
  // Stop the timer we started above
  fx_timer_stop(allnn_module, "tree_building");
  
  // An array for our output
  ArrayList<index_t> results;
  
  // Starting another timer
  fx_timer_start(allnn_module, "dual_tree_computation");
   
  // This function will do the work of actually computing the neighbors
  allnn.ComputeNeighbors(&results);
  
  // Stopping the timer above
  fx_timer_stop(allnn_module, "dual_tree_computation");
  
  // Specify a command line parameter to check the computation against naive
  int do_naive = fx_param_bool(NULL, "do_naive", 0);
    
  // if we want to do the naive check
  if (do_naive) {
    
    // create another module
    struct datanode* naive_module = fx_submodule(NULL, "naive", "naive_module");
    // create another object for it
    AllNN naive_allnn;
    // initialize the object, this time without building the trees
    naive_allnn.InitNaive(queries, references, naive_module);
  
    // results for the naive computation
    ArrayList<index_t> naive_results;

    // timing for the naive computation
    fx_timer_start(naive_module, "naive_time");
    
    // We'll just use the naive computation for this
    naive_allnn.ComputeNaive(&naive_results);
    
    // stopping the timer
    fx_timer_stop(naive_module, "naive_time");
    
    // Element by element, we'll compare the results of the dual-tree and naive computations, issuing a warning if they disagree
    for (index_t i = 0; i < results.size(); i++) {
      DEBUG_WARN_MSG_IF(results[i] != naive_results[i], "i = %d, results[i] = %d, naive_results[i] = %d", i, results[i], naive_results[i]);;
    }
    
  }
  
  // The name of the file we will write results to
  // Defaults to "output.csv"
  const char* output_filename = fx_param_str(NULL, "output_filename", "output.csv");
  
  // Need the file pointer for print
  FILE* output_file = fopen(output_filename, "w");
  
  // Will print results to the output file
  ot::Print(results, output_file);
  
  // This is terminates FASTexec.  Always put it at the end of main when using FASTexec.
  fx_done();
  
  // We're at the end of main
  return 0;
  
}