/*
 *  n_point_main.cc
 *  
 *
 *  Created by William March on 2/24/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "fastlib/fastlib.h"
#include "n_point_perm_free.h"
#include "n_point.h"


const fx_entry_doc n_point_main_entries[] = {
{"data", FX_REQUIRED, FX_STR, NULL, 
"Matrix containing the data points.\n"},
{"weights", FX_PARAM, FX_STR, NULL,
  "File with weights for the points.  If not specified, all weights are 1.\n"},
{"upper_bounds", FX_REQUIRED, FX_STR, NULL,
"File with upper bounds for the matcher.\n"},
{"lower_bounds", FX_PARAM, FX_STR, NULL,
"File with lower bounds for the matcher.  Assumed to be zero if unspecified.\n"},
{"do_naive", FX_PARAM, FX_BOOL, NULL,
  "Run a naive implementation, default: false\n"},
{"do_hybrid", FX_PARAM, FX_BOOL, NULL,
"Do the hybrid expansion pattern, default: false\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc n_point_main_submodules[] = {
{"naive", &n_point_doc,
  "Naive implementation module.\n"},
{"depth_first", &n_point_doc, 
  "Depth first recursion implementation module.\n"},
{"hybrid_expansion", &n_point_doc,
  "Hybrid expansion module.\n"},
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc n_point_main_doc = {
n_point_main_entries, n_point_main_submodules,
"Main for different n-point routines.\n"
};

///////////////////////////////////////////


int main(int argc, char* argv[]) {

  fx_init(argc, argv, &n_point_main_doc);

  /*
  factorials[0] = 1;
  for (int i = 1; i < MAX_FAC; i++) {
    factorials[i] = i * factorials[i-1];
  } // fill in factorials array
  */
  
  
  //////// read in data /////////////
  
  Matrix data;
  const char* data_file = fx_param_str_req(NULL, "data");
  data::Load(data_file, &data);
  
  Vector weights;
  if (fx_param_exists(NULL, "weights")) {
    Matrix weights_mat;
    const char* weights_file = fx_param_str_req(NULL, "weights");
    data::Load(weights_file, &weights_mat);
    weights.Copy(weights_mat.ptr(), data.n_rows());
  }
  else {
    weights.Init(data.n_rows());
    weights.SetAll(1.0);
  }
  
  Matrix upper_bounds;
  const char* upper_file = fx_param_str_req(NULL, "upper_bounds");
  data::Load(upper_file, &upper_bounds);
  
  Matrix lower_bounds;
  if (fx_param_exists(NULL, "lower_bounds")) {
    const char* lower_file = fx_param_str_req(NULL, "lower_bounds");
    data::Load(lower_file, &lower_bounds);    
  }
  else {
    lower_bounds.Init(upper_bounds.n_rows(), upper_bounds.n_cols());
    lower_bounds.SetAll(0.0);
  }
  
  // TODO: check input bounds
  
  /////////// run algorithm //////////////

  if (fx_param_exists(NULL, "do_naive")) {
  
    fx_module* naive_mod = fx_submodule(NULL, "naive");
    fx_set_param_bool(naive_mod, "do_naive", true);
    
    NPointAlg alg;
    alg.Init(data, weights, lower_bounds, upper_bounds, upper_bounds.n_cols(), 
             naive_mod);
    
    alg.Compute();
        
  }
  else if (fx_param_exists(NULL, "do_hybrid")) {
    
    fx_module* hybrid_mod = fx_submodule(NULL, "hybrid_expansion");
    fx_set_param_bool(hybrid_mod, "do_hybrid", true);
    
    NPointAlg alg;
    alg.Init(data, weights, lower_bounds, upper_bounds, upper_bounds.n_cols(), 
             hybrid_mod);
    
    alg.Compute();
    
    
  }
  else if (fx_param_exists(NULL, "do_perm_free")) {
    
    fx_module* perm_free_mod = fx_submodule(NULL, "perm_free");
    
    NPointPermFree perm_free_alg;
    
    perm_free_alg.Init(data, lower_bounds, upper_bounds, upper_bounds.n_cols(),
                       perm_free_mod);
    perm_free_alg.Compute();
    
  }
  else {
   
    fx_module* depth_mod = fx_submodule(NULL, "depth_first");
    
    
    
    NPointAlg alg;
    alg.Init(data, weights, lower_bounds, upper_bounds, upper_bounds.n_cols(), 
             depth_mod);
    
    alg.Compute();
    
  }
  
  
  fx_done(NULL);
  
  return 0;
  
} // main ()