/*
 *  n_point_multi_main.cc
 *  
 *
 *  Created by William March on 6/7/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */


#include "fastlib/fastlib.h"
#include "n_point_multi.h"

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
  
  Matrix data;
  const char* data_file = fx_param_str_req(NULL, "data");
  data::Load(data_file, &data);
  
  double min_band, max_band;
  min_band = fx_param_double_req(NULL, "min_band");
  max_band = fx_param_double_req(NULL, "max_band");
  int num_bands = fx_param_int_req(NULL, "num_bands");
  
  double band_width;
  
  int n = fx_param_int_req(NULL, "n");

  fx_module* mod = fx_submodule(NULL, "n_point_multi");
  
  NPointMulti alg;
  alg.Init(data, min_band, max_band, num_bands, n, mod);

  alg.Compute();
  
  fx_done(NULL);
  
  return 0;
  
} // main()