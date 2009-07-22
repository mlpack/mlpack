#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"
#include "thor_2point.h"

const fx_entry_doc root_entries[] = { 
  {"n", FX_PARAM, FX_INT, NULL,
   "Specifies number of points. \n"}, 
  FX_ENTRY_DOC_DONE  
};

const fx_submodule_doc npc_submodules[] = {
   {"param", &param_doc, "Parameters for 2 Point Correlation \n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc root_doc = {
  root_entries,  npc_submodules,
  "Computation Parameters \n"
};



int main(int argc, char *argv[]) {  
  fx_module *root = fx_init(argc, argv, &root_doc);

  Thor2PC sky;

   // Read in positions
  struct datanode* parameters = fx_submodule(root, "param");
  
  sky.Init(parameters);
  sky.Compute(parameters);
  Vector results, bins;
  sky.OutputResults(results);
  sky.GetBins(bins);

  // Write Results

  fx_done(fx_root);
  return 0;
}
