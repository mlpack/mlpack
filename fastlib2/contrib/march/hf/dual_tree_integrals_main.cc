#include "dual_tree_integrals.h"
#include "naive_fock_matrix.h"

int main(int argc, char* argv[]) {

  fx_init(argc, argv, NULL);

  printf("Doing just integrals\n");

  DualTreeIntegrals integrals;
  
  Matrix centers;
  const char* centers_name = fx_param_str_req(NULL, "basis_centers");
  data::Load(centers_name, &centers);
  
  double angstroms_to_bohr = 1.889725989;

  if (fx_param_exists(NULL, "angstroms")) {
    la::Scale(angstroms_to_bohr, &centers);
  }

  struct datanode* mod = fx_submodule(NULL, "integrals");
  
  double bandwidth = fx_param_double(NULL, "bandwidth", 0.1);
  
  integrals.Init(centers, mod, bandwidth);
  
  Matrix density;
  const char* density_name = fx_param_str(NULL, "density", "");
  
  if (SUCCESS_PASS != data::Load(density_name, &density)) {
    density.Destruct();
    density.Init(centers.n_cols(), centers.n_cols());
    density.SetAll(1.0);
  }
  
  integrals.SetDensity(density);
  
  fx_timer_start(NULL, "nbody_time");
  integrals.ComputeFockMatrix();
  fx_timer_stop(NULL, "nbody_time");
  
  Matrix fock_out;
  integrals.OutputFockMatrix(&fock_out, NULL, NULL, NULL);

  fock_out.PrintDebug();
  
  if (fx_param_exists(NULL, "do_naive")) {
    
    struct datanode* naive_mod = fx_submodule(NULL, "naive");
  
    NaiveFockMatrix naive;
    naive.Init(centers, naive_mod, density, bandwidth);
    
    fx_timer_start(NULL, "naive_time");
    naive.ComputeFockMatrix();
    fx_timer_stop(NULL, "naive_time");
    
    Matrix naive_fock;
    naive.PrintFockMatrix(&naive_fock, NULL, NULL);
  
    naive_fock.PrintDebug();
  
  }

  fx_done(NULL);

  return 0;

} // main()
