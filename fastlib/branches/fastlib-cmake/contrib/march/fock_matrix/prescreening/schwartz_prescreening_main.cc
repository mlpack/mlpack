#include "fastlib/fastlib.h"
#include "schwartz_prescreening.h"
//#include "eri.h"



// I'm assuming that there is a vector in centers for each shell, i.e. if a 
// carbon atom has 2s and three p functions, then the center will appear 3 
// times, once for each s along with an exponent and momentum entry of 0, and 
// once for the p shell with exponents and momentum entry of 1
// Therefore, the ultimate size of the fock matrix should be given by 
// centers.n_cols() + 2*sum(momenta)

const fx_entry_doc schwartz_prescreening_entries[] = {
  {"centers", FX_PARAM, FX_STR, NULL, 
   "A file containing the centers of the basis functions.  \n"
   "There should be one entry per s or p primitive. \n"
      "(defaults to test_centers.csv)\n"},
  {"threshold", FX_PARAM, FX_DOUBLE, NULL, "The cutoff threshold (default: 10^-10)\n"},
  {"density", FX_PARAM, FX_STR, NULL, 
   "A file containing the input density matrix.\n"},
  {"fock_matrix", FX_TIMER, FX_DOUBLE, NULL, 
    "Total time to create shell pairs and compute fock matrix.\n"}, 
   FX_ENTRY_DOC_DONE
};

const fx_submodule_doc schwartz_prescreening_submodules[] = {
  {"schwartz", &schwartz_mod_doc, "algorithm submodule"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc schwartz_prescreening_main_doc = {
  schwartz_prescreening_entries, schwartz_prescreening_submodules, 
     "Schwartz bound prescreening as described in Haser's paper.\n"
};

int main(int argc, char* argv[]) {

  // fill this in some time
 
  fx_module* root_mod = fx_init(argc, argv, &schwartz_prescreening_main_doc);
  
  
  // Load data
  const char* centers_file = fx_param_str(root_mod, "centers", "test_centers.csv");
  Matrix centers;
  data::Load(centers_file, &centers);

  double angstroms_to_bohr = 1.889725989;

  if (fx_param_exists(root_mod, "angstroms")) {
    
    la::Scale(angstroms_to_bohr, &centers);

  }  
  
  // Change these to believe input later
  Vector exp;
  if (fx_param_exists(root_mod, "single_bandwidth")) {
    double band = fx_param_double(root_mod, "single_bandwidth", 0.01);
    exp.Init(centers.n_cols());
    exp.SetAll(band);
  }
  else {
    const char* exp_file = fx_param_str(root_mod, "exponents", "test_exp.csv");
    Matrix exponents;
    data::Load(exp_file, &exponents);
    exponents.MakeColumnVector(0, &exp);
  }

  

  /*
  const char* momenta_file = fx_param_str(root_mod, "momenta", "test_momenta.csv");
  Matrix momenta;
  data::Load(momenta_file, &momenta);
  */
  Vector mom;
  mom.Init(centers.n_cols());
  mom.SetAll(0);
  
  // is equal to n_cols + 2*sum(mom) for only s and p-type
  index_t mat_size;
  // The dot product only works for momenta 1 
  mat_size = centers.n_cols() + (index_t)2*la::Dot(mom, mom);
  
  Matrix density;
  if (fx_param_exists(root_mod, "density")) {
    const char* density_file = fx_param_str_req(root_mod, "density");
    data::Load(density_file, &density);
  }
  else {
    density.Init(mat_size, mat_size);
  }
  
  double thresh = fx_param_double(root_mod, "threshold", 10e-10);
  
  fx_module* schwartz_mod = fx_submodule(root_mod, "schwartz");
  
  // Needs a density matrix
  
  SchwartzPrescreening screen;
  screen.Init(centers, exp, mom, thresh, density, mat_size, schwartz_mod);
  
  Matrix fock_mat;
  
  fx_timer_start(root_mod, "fock_matrix");
  screen.ComputeFockMatrix(&fock_mat);
  fx_timer_stop(root_mod, "fock_matrix");
  
  
  
  fx_done(root_mod);


  return 0;

} // main()
