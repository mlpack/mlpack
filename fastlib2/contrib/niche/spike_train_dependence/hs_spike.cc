#include "fastlib/fastlib.h"
#include "hs_spike.h"
#include "hsic.h"




int main(int argc, char* argv[]) {
  
  fx_module* root = fx_init(argc, argv, NULL);
  
  const char* x_filename = fx_param_str(NULL, "x", "x.csv");
  const char* y_filename = fx_param_str(NULL, "y", "y.csv");

  const int dependence_horizon = fx_param_int(NULL, "tau", 2);

  SpikeSeqPair spikes_pair;
  spikes_pair.Init(x_filename, y_filename, dependence_horizon);

  //spikes_pair.Merge();
  //spikes_pair.PrintAllSpikes();

  Matrix x1, y1, x2, y2;
  spikes_pair.ConstructPoints(&x1, &y1, &x2, &y2);

  double hsic_x1_y1 = HSIC(x1, y1);
  printf("x reference hsic(x, y) = %f\n", hsic_x1_y1);

  double hsic_x2_y2 = HSIC(x2, y2);
  printf("y reference hsic(x, y) = %f\n", hsic_x2_y2);


  fx_done(root);

}
