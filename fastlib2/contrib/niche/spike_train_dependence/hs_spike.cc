#include "fastlib/fastlib.h"
#include "hs_spike.h"

int main(int argc, char* argv[]) {
  
  fx_module* root = fx_init(argc, argv, NULL);
  
  const char* x_filename = fx_param_str(NULL, "x", "x.csv");
  const char* y_filename = fx_param_str(NULL, "y", "y.csv");

  const int dependence_horizon = fx_param_int(NULL, "tau", 2);

  SpikeSeqPair spikes_pair;
  spikes_pair.Init(x_filename, y_filename, dependence_horizon);

  spikes_pair.Merge();
  spikes_pair.PrintAllSpikes();
  spikes_pair.ConstructPoints();
  //spikes_pair.XRef();

  
  

  fx_done(root);

}
