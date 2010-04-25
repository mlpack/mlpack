#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"
#include "contrib/march/fock_matrix/prescreening/schwartz_prescreening.h"


int main(int argc, char* argv[]) {

  fx_module* root_mod = fx_init(argc, argv, NULL);

  const char* centers_file = fx_param_str(root_mod, 
                                          "centers", "test_centers.csv");
                                          
  Matrix centers;
  
  data::Load(centers_file, &centers);
  
  Vector mom;
  mom.Init(centers.n_cols());
  mom.SetAll(0);
  
  index_t mat_size;
  mat_size = centers.n_cols() + 2*la::Dot(mom, mom);
  
  Matrix density;
  if (fx_param_exists(root_mod, "density")) {
    const char* density_file = fx_param_str_req(root_mod, "density");
    data::Load(density_file, &density);
  }
  else {
    density.Init(mat_size, mat_size);
    density.SetAll(1.0);
  }
  
  double common_exp;
  common_exp = fx_param_double(root_mod, "exponent", 0.01);
  
  Vector exponents;
  exponents.Init(centers.n_cols());
  exponents.SetAll(common_exp);
  
  double thresh = fx_param_double(root_mod, "threshold", 10e-10);

  fx_module* naive_mod = fx_submodule(root_mod, "naive");

  NaiveFockMatrix naive_computation;
  naive_computation.Init(centers, naive_mod, density, common_exp);
  
  fx_timer_start(root_mod, "naive_time");
  naive_computation.ComputeFock();
  fx_timer_stop(root_mod, "naive_time");

  Matrix naive_fock;
  naive_computation.OutputFock(&naive_fock, NULL, NULL);
  
  printf("\nNaiveFock.\n");
  naive_fock.PrintDebug();


  //////////////
  
  fx_module* schwartz_mod = fx_submodule(root_mod, "schwartz");
  
  SchwartzPrescreening screened_computation;
  screened_computation.Init(centers, exponents, mom, thresh, density, mat_size, 
                            schwartz_mod);
  
  
  Matrix schwartz_fock;
  
  fx_timer_start(root_mod, "screened_time");
  screened_computation.ComputeFockMatrix(&schwartz_fock);
  fx_timer_stop(root_mod, "screened_time");
  
  printf("\nSchwartz Fock\n");
  schwartz_fock.PrintDebug();
  
  //////////////////
  
  Matrix diff;
  la::SubInit(naive_fock, schwartz_fock, &diff);
  
  diff.PrintDebug();
  
  
  
  fx_done(root_mod);


  return 0;

} // main()