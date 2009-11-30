/*
 *  comparison_main.cc
 *  
 *
 *  Created by William March on 10/20/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "mst_comparison.h"
#include "dtb_cover.h"
#include "mlpack/emst/dtb.h"


int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);
  
  const char* data_name = fx_param_str_req(NULL, "data");
  
  Matrix data;
  data::Load(data_name, &data);
  
  DualTreeBoruvka dtb;
  //struct datanode* dtb_module = fx_submodule(NULL, "dtb", "dtb_module");
  struct datanode* dtb_module = fx_submodule(NULL, "dtb_module");
  dtb.Init(data, dtb_module);
  
  ////////////// Run DTB /////////////////////
  Matrix kd_results;
  
  dtb.ComputeMST(&kd_results);
  
  kd_results.PrintDebug("kd-tree results");

  
  DualCoverTreeBoruvka dtb_cover;
  fx_module* cover_module = fx_submodule(NULL, "cover_module");
  dtb_cover.Init(data, cover_module);
  
  Matrix cover_results;
  dtb_cover.ComputeMST(&cover_results);
  
  cover_results.PrintDebug("cover results");
  
  MSTComparison compare;
  compare.Init(kd_results, cover_results);
  
  bool same = compare.Compare();
  
  if (same) {
    printf("MST's are equal.\n");
  }
  
  
  fx_done(NULL);
  
  return 0;
  
} 