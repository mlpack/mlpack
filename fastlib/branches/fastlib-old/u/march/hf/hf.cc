#include "u/dongrel/multibody.h"
#include "u/dongrel/multibody_kernel.h"
#include "hf_kernel.h"

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv);
  
  /* for now, assuming global fixed bandwidth */
  double bandwidth;
  bandwidth = fx_param_double(NULL, "bandwidth", 0.1);
  
  /* not sure what this means yet */
  double tau;
  tau = fx_param_double(NULL, "tau", 0.1);
  
  /* to add, check for basic HF vs. multitree HF, implement both */
  
  printf("Starting multibody HF\n");
  fx_timer_start(NULL, "multibodyHF");
  
  MultitreeMultibody<GaussianThreeBodyKernel, GaussianKernel, GaussianKernelDerivative> mthf;
  mthf.Init(bandwidth);
  mthf.Compute(tau);
  fx_timer_stop(NULL, "multibodyHF");
  printf("Multibody HF complete\n");
  
  
  fx_done();
  
}