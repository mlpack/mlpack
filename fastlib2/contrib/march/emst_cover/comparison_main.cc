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

const fx_entry_doc comparison_main_entries[] = {
  {"data", FX_REQUIRED, FX_STR, NULL,
  "A file containing the data.\n"},
};

const fx_submodule_doc comparison_main_submodules[] = {
{"cover_module", &dtb_cover_doc,
  "Cover tree module.\n"},
{"dtb_module", &dtb_doc,
  "kd-tree module.\n"},
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc comparison_main_doc = {
  comparison_main_entries, comparison_main_submodules,
  "Runs and compares different methods for computing MST's.\n"
};

int main(int argc, char* argv[]) {

  fx_init(argc, argv, &comparison_main_doc);
  
  const char* data_name = fx_param_str_req(NULL, "data");
  
  Matrix data;
  data::Load(data_name, &data);
  
  DualTreeBoruvka dtb;
  //struct datanode* dtb_module = fx_submodule(NULL, "dtb", "dtb_module");
  struct datanode* dtb_module = fx_submodule(NULL, "dtb_module");
  dtb.Init(data, dtb_module);
  
  ////////////// Run DTB /////////////////////
  Matrix kd_results;
  
  printf("===== Computing kd-tree results =====\n");
  dtb.ComputeMST(&kd_results);
  
  //kd_results.PrintDebug("kd-tree results");

  
  DualCoverTreeBoruvka dtb_cover;
  fx_module* cover_module = fx_submodule(NULL, "cover_module");
  dtb_cover.Init(data, cover_module);
  
  printf("\n===== Computing cover tree results =====\n");
  
  Matrix cover_results;
  dtb_cover.ComputeMST(&cover_results);
  
  //cover_results.PrintDebug("cover results");
  
  MSTComparison compare;
  compare.Init(kd_results, cover_results);
  
  bool same = compare.Compare();
  
  if (same) {
    printf("\n====== MST's are equal. ======\n\n");
  }
  
  
  fx_done(NULL);
  
  return 0;
  
} 