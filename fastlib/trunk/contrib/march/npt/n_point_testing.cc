/*
 *  n_point_testing.cc
 *  
 *
 *  Created by William March on 2/16/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "naive_two_point.h"
#include "fastlib/fastlib.h"

const fx_entry_doc n_point_testing_main_entries[] = {
{"data", FX_REQUIRED, FX_STR, NULL,
"A file containing the data.\n"},
};

const fx_submodule_doc n_point_testing_main_submodules[] = {
{"naive_two_point_module", &naive_two_point_doc,
"Naive two-point module.\n"},
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc n_point_testing_main_doc = {
n_point_testing_main_entries, n_point_testing_main_submodules,
"Runs and compares different methods for computing n point correlations.\n"
};

int main(int argc, char* argv[]) {
  
  fx_init(argc, argv, &n_point_testing_main_doc);
  
  const char* data_name = fx_param_str_req(NULL, "data");
  
  Matrix data;
  data::Load(data_name, &data);
  
  //////////// Naive two point //////////////////
  
  fx_module* naive_two_point_mod = fx_submodule(NULL, "naive_two_point_module");
  
  NaiveTwoPoint naive_two_point_alg;
  naive_two_point_alg.Init(data, naive_two_point_mod);

  naive_two_point_alg.Compute();
  
  fx_done(NULL);
  
  return 0;
  
} 
