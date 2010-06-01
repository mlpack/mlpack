#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"
#include "thor_3point.h"

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
  rpc::Init();

  const char* fp_results;
  FILE *results;
  Thor3PC sky;

   // Read in positions
  struct datanode* parameters = fx_submodule(root, "param");
  
  sky.Init(parameters);
  sky.Compute(parameters);
  Vector counts, bins;
  sky.OutputResults(counts);
  sky.GetBins(bins);
  ArrayList<ArrayList<Vector> > count3;
  sky.OutputResults3(count3);

  if (rpc::is_root()){
    // Write Results
    fp_results = fx_param_str(NULL, "output", "results.dat");
    results = fopen(fp_results, "w+");
    fprintf(results,"Start \t Stop \t Count \n");
    for (int i = 0; i< counts.length(); i++){
      fprintf(results, "%7.6f \t %7.6f \t %8.1f \n", sqrt(bins[i]), 
	      sqrt(bins[i+1]),  counts[i]);
    }    
  }

  if (rpc::is_root()){
    // Write Results
    fp_results = fx_param_str(NULL, "output3", "results3.dat");
    results = fopen(fp_results, "w+");
    fprintf(results,"Start \t Stop \t Count \n");
    for(int i = 0; i< counts.length(); i++){
      for(int j = 0; j < counts.length(); j++){
	for(int k = 0; k < counts.length(); k++){
	  fprintf(results, "%4.3f \t %4.3f \t ", sqrt(bins[i]),
		  sqrt(bins[i+1]));
	  fprintf(results, "%4.3f \t %4.3f \t ", sqrt(bins[j]),
		  sqrt(bins[j+1]));
	  fprintf(results, "%4.3f \t %4.3f \t ", sqrt(bins[k]),
		  sqrt(bins[k+1]));
	  fprintf(results, "%8.1f \n", ((count3[i])[j])[k]);
	}
      }
    }    
  }


  rpc::Done();
  fx_done(fx_root);
  return 0;
}
