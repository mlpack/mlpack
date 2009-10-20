/**
* @file emst_cover_main.cc
 *
 * Cover tree implementation of DualTreBoruvka algorithm.
 *
 * @author Bill March (march@gatech.edu)
*/

#include "dtb_cover.h"

int main(int argc, char* argv[]) {
 
  fx_init(argc, argv, NULL);
 
  // For when I implement a thor version
  bool using_thor = fx_param_bool(NULL, "using_thor", 0);
  
  
  if unlikely(using_thor) {
    printf("thor is not yet supported\n");
  }
  else {
      
    ///////////////// READ IN DATA ////////////////////////////////// 
    
    const char* data_file_name = fx_param_str_req(NULL, "data");
    
    Matrix data_points;
    
    data::Load(data_file_name, &data_points);
    
    /////////////// Initialize DTB //////////////////////
    DualCoverTreeBoruvka dtb;
    struct datanode* dtb_module = fx_submodule(NULL, "dtb_module");
    dtb.Init(data_points, dtb_module);
    
    ////////////// Run DTB /////////////////////
    Matrix results;
    
    dtb.ComputeMST(&results);
    
    //results.PrintDebug("Results");
    
    //////////////// Check against naive //////////////////////////
#if 0
    if (fx_param_bool(NULL, "do_naive", 0)) {
     
      DualTreeBoruvka naive;
      struct datanode* naive_module = fx_submodule(NULL, "naive_module");
      fx_set_param_bool(naive_module, "do_naive", 1);
      
      naive.Init(data_points, naive_module);
      
      Matrix naive_results;
      naive.ComputeMST(&naive_results);
      
      MSTComparison compare;
      compare.Init(results, naive_results);
      
      bool same = compare.Compare();
      
      if (same) {
        printf("MST's are equal.\n");
      }
            
    } // doing naive
#endif
    //////////////// Output the Results ////////////////
    
    const char* output_filename = 
        fx_param_str(NULL, "output_filename", "output.txt");
    
    //FILE* output_file = fopen(output_filename, "w");
    
    data::Save(output_filename, results);
    
  }// end else (if using_thor)
  
  fx_done(NULL);
  
  return 0;
  
}