/**
* @file emst_cover_main.cc
 *
 * Cover tree implementation of DualTreBoruvka algorithm.
 *
 * @author Bill March (march@gatech.edu)
*/

#include "dtb_cover.h"
#include "mlpack/emst/dtb.h"
#include "geomst2.h"
#include "friedman_bentley.h"
#include "multi_fragment.h"

const fx_entry_doc cover_main_entries[] = {
{"data", FX_REQUIRED, FX_STR, NULL,
"A file containing the data.\n"},
{"do_cover", FX_PARAM, FX_BOOL, NULL,
  "Specify to run the cover tree DTB algorithm.\n"},
{"do_naive", FX_PARAM, FX_BOOL, NULL,
"Specify to run the naive Boruvka's algorithm.\n"},
{"do_kd", FX_PARAM, FX_BOOL, NULL,
"Specify to run the kd-tree DTB algorithm.\n"},
{"do_geomst", FX_PARAM, FX_BOOL, NULL,
"Specify to run the GeoMST2 algorithm.\n"},
{"do_single_fragment", FX_PARAM, FX_BOOL, NULL,
  "Specify to run the Friedman-Bentley single fragment algorithm\n"},
{"do_multi_fragment", FX_PARAM, FX_BOOL, NULL,
"Specify to run the NEKMA multi fragment algorithm\n"},
{"using_thor", FX_PARAM, FX_BOOL, NULL,
  "Not yet supported, do not specify.\n"},
};

const fx_submodule_doc cover_main_submodules[] = {
{"cover_module", &dtb_cover_doc,
"Cover tree module.\n"},
{"kd_module", &dtb_doc,
"kd-tree module.\n"},
{"naive_module", &dtb_doc, "naive module\n"},
{"geo_module", &geomst_doc, "geomst module\n"},
{"single_fragment_module", &fb_doc, "single fragment module.\n"},
{"multi_fragment_module", &mf_doc, "multi fragment module\n"},
FX_SUBMODULE_DOC_DONE
};

const fx_module_doc cover_main_doc = {
cover_main_entries, cover_main_submodules,
"Runs different methods for computing MST's.\n"
};



int main(int argc, char* argv[]) {
 
  fx_init(argc, argv, &cover_main_doc);
 
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
    
    //////////// Cover Tree //////////////
    if (fx_param_exists(NULL, "do_cover")) {
    
    /////////////// Initialize DTB //////////////////////
    DualCoverTreeBoruvka dtb;
    struct datanode* dtb_module = fx_submodule(NULL, "cover_module");
    dtb.Init(data_points, dtb_module);
    
    ////////////// Run DTB /////////////////////
    Matrix results;
    
    dtb.ComputeMST(&results);
    
    //results.PrintDebug("Results");

    }
    
    //////////// Naive //////////////
    if (fx_param_exists(NULL, "do_naive")) {
      
      /////////////// Initialize DTB //////////////////////
      DualTreeBoruvka naive;
      struct datanode* naive_module = fx_submodule(NULL, "naive_module");
      fx_set_param_bool(naive_module, "do_naive", 1);
      naive.Init(data_points, naive_module);
      
      ////////////// Run DTB /////////////////////
      Matrix results;
      
      naive.ComputeMST(&results);
      
      //results.PrintDebug("Results");
      
    }

    //////////// kd-tree //////////////
    if (fx_param_exists(NULL, "do_kd")) {
      
      /////////////// Initialize DTB //////////////////////
      DualTreeBoruvka kd;
      struct datanode* kd_module = fx_submodule(NULL, "kd_module");
      kd.Init(data_points, kd_module);
      
      ////////////// Run DTB /////////////////////
      Matrix results;
      
      kd.ComputeMST(&results);
      
      //results.PrintDebug("Results");
      
    }
    
    //////////// geomst //////////////
    if (fx_param_exists(NULL, "do_geomst")) {
      
      Matrix geomst_results;
      
      GeoMST2 geo;
      fx_module* geo_mod = fx_submodule(NULL, "geo_module");
      
      geo.Init(data_points, geo_mod);
      
      geo.ComputeMST(&geomst_results);
      
      //results.PrintDebug("Results");
      
    }
    
    ///////// Single Fragment Friedman-Bentley algorithm ///////
    if (fx_param_exists(NULL, "do_single_fragment")) {
      
      Matrix single_fragment_results;
      
      FriedmanBentley fb_alg;
      fx_module* fb_mod = fx_submodule(NULL, "single_fragment_module");
      
      fb_alg.Init(data_points, fb_mod);
      fb_alg.ComputeMST(&single_fragment_results);
      
    }
    
    /////////// Multi Fragment (NEKMA) algorithm //////////////
    if (fx_param_exists(NULL, "do_multi_fragment")) {
      
      Matrix multi_fragment_results;
      
      MultiFragment mf_alg;
      fx_module* mf_mod = fx_submodule(NULL, "multi_fragment_module");
      
      mf_alg.Init(data_points, mf_mod);
      mf_alg.ComputeMST(&multi_fragment_results);
      
    }
    
    
    
    
    
    
  }// end else (if using_thor)
  
  fx_done(NULL);
  
  return 0;
  
}