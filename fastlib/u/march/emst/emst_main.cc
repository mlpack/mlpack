/**
* @file emst.cc
 *
 * Calls the DualTreeBoruvka algorithm from dtb.h
 * Can optionally call Naive Boruvka's method
 * See README for command line options.  
 *
 * @author Bill March (march@gatech.edu)
*/

#include "dtb.h"


int main(int argc, char* argv[]) {
 
  fx_init(argc, argv);
 
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
    DualTreeBoruvka dtb;
    struct datanode* dtb_module = fx_submodule(NULL, "dtb", "dtb_module");
    dtb.Init(data_points, dtb_module);
    
    ////////////// Run DTB /////////////////////
    ArrayList<EdgePair> results;
    
    dtb.ComputeMST(&results);
    
    //////////////// Check against naive //////////////////////////
    if (fx_param_bool(NULL, "do_naive", 0)) {
     
      DualTreeBoruvka naive;
      struct datanode* naive_module = fx_submodule(NULL, "naive", 
                                                   "naive_module");
      fx_set_param(naive_module, "do_naive", "1");
      
      naive.Init(data_points, naive_module);
      
      ArrayList<EdgePair> naive_results;
      naive.ComputeMST(&naive_results);
      
      fx_timer_start(naive_module, "comparison");
      
      if (fx_get_result_double(dtb_module, "total_squared_length") !=
          fx_get_result_double(naive_module, "total_squared_length")) { 
       
        printf("lengths are different!\n");
        
        fx_done();
        return 1;
        
      }
      else {
        printf("lengths are the same\n");
      }
      
      int is_correct = 1;
      for (index_t naive_index = 0; naive_index < results.size(); 
           naive_index++) {
       
        int this_loop_correct = 0;
        index_t naive_lesser_index = results[naive_index].lesser_index();
        index_t naive_greater_index = results[naive_index].greater_index();
        double naive_distance = results[naive_index].distance();
        
        for (index_t dual_index = 0; dual_index < naive_results.size();
             dual_index++) {
          
          index_t dual_lesser_index = results[dual_index].lesser_index();
          index_t dual_greater_index = results[dual_index].greater_index();
          double dual_distance = results[dual_index].distance();
          
          if (naive_lesser_index == dual_lesser_index) {
            if (naive_greater_index == dual_greater_index) {
              DEBUG_ASSERT(naive_distance == dual_distance);
              this_loop_correct = 1;
              break;
            }
          }
          
        }
       
        if (this_loop_correct == 0) {
          is_correct = 0;
          break;
        }
        
      }
      
      if (is_correct == 0) {
       
        printf("naive check failed!\n");
        if (fx_get_result_double(dtb_module, "total_squared_length") !=
            fx_get_result_double(naive_module, "total_squared_length")) { 
          
          printf("lengths are different!\n");
          
          fx_done();
          return 1;
          
        }
        else {
          printf("lengths are the same\n");
        }
      
      }
      
      fx_timer_stop(naive_module, "comparison");
      
      const char* naive_output_filename = 
        fx_param_str(naive_module, "output_filename", "naive_output.txt");
      
      FILE* naive_output = fopen(naive_output_filename, "w");
      
      ot::Print(naive_results, naive_output);
      
    }
    
    //////////////// Output the Results ////////////////
    
    const char* output_filename = 
        fx_param_str(NULL, "output_filename", "output.txt");
    
    FILE* output_file = fopen(output_filename, "w");
    
    ot::Print(results, output_file);
    
  }// end else (if using_thor)
  
  fx_done();
  
  return 0;
  
}