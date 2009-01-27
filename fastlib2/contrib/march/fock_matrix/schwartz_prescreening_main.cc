#include "fastlib/fastlib.h"
#include "schwartz_prescreening.h"
#include "eri.h"


int main(int argc, char* argv[]) {

  // fill this in some time
  fx_module_doc* doc = NULL;

  fx_module* root_mod = fx_init(argc, argv, doc);
  
  
  // Load data
  const char* centers_file = fx_param_str(root_mod, "centers", "test_centers.csv");
  Matrix centers;
  data::Load(centers_file, &centers);
  
  const char* exp_file = fx_param_str(root_mod, "exponents", "test_exp.csv");
  Matrix exponents;
  data::Load(exp_file, &exponents);
  Vector exp;
  exponents.MakeColumnVector(0, &exp);
  
  const char* momenta_file = fx_param_str(root_mod, "momenta", "test_momenta.csv");
  Matrix momenta;
  data::Load(momenta_file, &momenta);
  
  // Change this to believe input later
  Vector mom;
  mom.Init(centers.n_cols());
  mom.SetAll(1);
  
  
  double thresh = fx_param_double(root_mod, "threshold", 10e-10);
  
  fx_module* schwartz_mod = fx_submodule(root_mod, "schwartz");
  
  // Needs a density matrix
  
  SchwartzPrescreening screen;
  screen.Init(centers, exp, mom, thresh, schwartz_mod);
  
  Matrix fock_mat;
  
  fx_timer_start(root_mod, "fock_matrix");
  screen.ComputeFockMatrix(&fock_mat);
  fx_timer_stop(root_mod, "fock_matrix");
  
  
  
  fx_done(root_mod);


  return 0;

} // main()