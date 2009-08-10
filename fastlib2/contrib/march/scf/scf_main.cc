#include "fastlib/fastlib.h"
#include "scf_solver.h"
//#include "contrib/march/fock_matrix/multi_tree/multi_tree_fock.h"
#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"
//#include "contrib/march/fock_matrix/prescreening/schwartz_prescreening.h"
//#include "contrib/march/fock_matrix/link/link.h"
//#include "contrib/march/fock_matrix/cfmm/cfmm_coulomb.h"



const fx_entry_doc scf_main_entries[] = {
{"centers", FX_REQUIRED, FX_STR, NULL, 
"A file containing the centers of the basis functions.\n"},
{"bohr", FX_PARAM, FX_STR, NULL, 
"Specify this parameter if the data are in bohr.  Otherwise they are assumed\n"
" to be in angstroms.\n"},
{"exponents", FX_REQUIRED, FX_STR, NULL, 
"A file containing the exponents of the basis functions.\n"
"Must have the same number of rows as centers.\n"},
{"density", FX_PARAM, FX_STR, NULL, 
"A file containing the density matrix.  If it is not provided, an all-ones\n"
"matrix is assumed.\n"},
{"momenta", FX_PARAM, FX_STR, NULL, 
"A file containing the momenta.  If not specified, then all functions are\n"
"assumed to be s-type.\n"},
{"density", FX_PARAM, FX_STR, NULL,
"A file containing the initial guess for the density.  If not specified, then\n"
"the all-zero density is used.\n"},
{"nuclear_centers", FX_REQUIRED, FX_STR, NULL,
"A file containing the locations of the nuclei.  Required.\n"},
{"nuclear_charges", FX_PARAM, FX_STR, NULL,
"A file containing the charges of the nuclei.  If not specified, then all\n"
"atoms are assumed to be H.\n"},
{"num_electrons", FX_REQUIRED, FX_INT, NULL,
"The total number of electrons.  Must be even, since only RHF supported.\n"},
{"coulomb_alg", FX_PARAM, FX_STR, NULL,
"Specify an algorithm to compute the Coulomb matrix:\n"
"\tnaive, prescreening, cfmm, multi.  Default: naive\n"},
{"exchange_alg", FX_PARAM, FX_STR, NULL,
"Specify an algorithm to compute the exchange matrix:\n"
"\tnaive, prescreening, link, multi.  Default: naive\n"},
FX_ENTRY_DOC_DONE
};

const fx_submodule_doc scf_main_submodules[] = {
//{"cfmm", &cfmm_mod_doc, 
//"Parameters and results for the CFMM.\n"},
//{"link", &link_mod_doc,
//"Parameters and results for LinK.\n"},
//{"prescreening", &prescreening_mod_doc,
//"Parameters and results for Schwartz prescreening.\n"},
{"naive", &naive_mod_doc,
"Parameters and results for naive.\n"},
//{"multi", &multi_mod_doc,
//"Parameters and results for multi-tree algorithm.\n"},
{"solver", &scf_mod_doc, 
"Parameters and results for the SCF main routine.\n"},
FX_SUBMODULE_DOC_DONE
};


const fx_module_doc scf_main_doc = {
scf_main_entries, scf_main_submodules, 
"Runs and compares different fock matrix construction methods.\n"
};


int main(int argc, char* argv[]) {

  fx_module* root_mod = fx_init(argc, argv, &scf_main_doc);
  eri::ERIInit();
  
  
  Matrix centers;
  const char* centers_file = fx_param_str_req(root_mod, "centers");
  data::Load(centers_file, &centers);
  
  Matrix exp_mat;
  const char* exp_file = fx_param_str_req(root_mod, "exponents");
  data::Load(exp_file, &exp_mat);
  
  if (centers.n_cols() != exp_mat.n_cols()) {
    FATAL("Number of basis centers must equal number of exponents.\n");
  }
  
  Matrix momenta;
  if (fx_param_exists(root_mod, "momenta")) {
    const char* momenta_file = fx_param_str_req(root_mod, "momenta");
    data::Load(momenta_file, &momenta);
  }
  else {
    momenta.Init(1, centers.n_cols());
    momenta.SetAll(0);
    printf("Assuming all s-type functions.\n\n");
  }

  // WARNING: this hack only works for s and p functions
  index_t num_funs = centers.n_cols() + 2 * (index_t)la::Dot(momenta, momenta);
  
  Matrix density;
  if (fx_param_exists(root_mod, "density")) {
    const char* density_file = fx_param_str_req(root_mod, "density");
    data::Load(density_file, &density);
  }
  else {
    density.Init(num_funs, num_funs);
    density.SetZero();
    printf("\nUsing only core integrals for initial matrix.\n\n");
  }
  
  if ((density.n_cols() != num_funs) || 
      (density.n_rows() != num_funs)) {
    FATAL("Density matrix has wrong dimensions.\n");
  }
  
      
  Matrix nuclear_centers;
  const char* nuclear_centers_file = fx_param_str_req(root_mod, 
                                                      "nuclear_centers");
  data::Load(nuclear_centers_file, &nuclear_centers);
  
  Matrix nuclear_charges;
  if (fx_param_exists(root_mod, "nuclear_charges")) {
    const char* nuclear_charges_file = fx_param_str_req(root_mod, 
                                                        "nuclear_charges");
    data::Load(nuclear_charges_file, &nuclear_charges);
  }
  else {
    nuclear_charges.Init(1, nuclear_centers.n_cols());
    nuclear_charges.SetAll(1.0);
    printf("Assuming all H atoms.\n\n");
  }
  
  if (nuclear_centers.n_cols() != nuclear_charges.n_cols()) {
    FATAL("Must provide a charge for every nuclear center!\n");
  }
  
  int num_electrons = fx_param_int_req(root_mod, "num_electrons");
  if (num_electrons % 2 != 0) {
    FATAL("Only RHF supported -- must specify an even number of electrons.\n");
  }
  
  const double angstrom_to_bohr = 1.889725989;
  // if the data are not input in bohr, assume they are in angstroms
  if (!fx_param_exists(root_mod, "bohr")) {
    
    la::Scale(angstrom_to_bohr, &centers);
    la::Scale(angstrom_to_bohr, &nuclear_centers);
    
  }  
  
  const char* coulomb_alg_str = fx_param_str(root_mod, "coulomb_alg", "naive");
  const char* exchange_alg_str = fx_param_str(root_mod, "exchange_alg", "naive");
  
  fx_module* solver_mod = fx_submodule(root_mod, "solver");
  
  // make cases for SCFSolver
  // have to make calls to it here to keep it in scope
  /*
  if (strcmp(coulomb_alg_str, "cfmm") == 0 
      && strcmp(exchange_alg_str, "link") == 0) {
  
    CFMMCoulomb cfmm_alg;
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    cfmm_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);

    Link link_alg;
    fx_module* link_mod = fx_submodule(root_mod, "link");
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);

    SCFSolver<CFMMCoulomb, Link> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &cfmm_alg, &link_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "cfmm") == 0 
           && strcmp(exchange_alg_str, "prescreening") == 0) {
    
    CFMMCoulomb cfmm_alg;
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    cfmm_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);

    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    SCFSolver<CFMMCoulomb, SchwartzPrescreening> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &cfmm_alg, &prescreening_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "cfmm") == 0 
           && strcmp(exchange_alg_str, "multi") == 0) {
    
    CFMMCoulomb cfmm_alg;
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    cfmm_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);

    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);
    
    SCFSolver<CFMMCoulomb, MultiTreeFock> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &cfmm_alg, &multi_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "cfmm") == 0 
           && strcmp(exchange_alg_str, "naive") == 0) {
    
    CFMMCoulomb cfmm_alg;
    fx_module* cfmm_mod = fx_submodule(root_mod, "cfmm");
    cfmm_alg.Init(centers, exp_mat, momenta, density, cfmm_mod);

    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    SCFSolver<CFMMCoulomb, NaiveFockMatrix> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &cfmm_alg, &naive_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "prescreening") == 0 
           && strcmp(exchange_alg_str, "link") == 0) {
    
    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    Link link_alg;
    fx_module* link_mod = fx_submodule(root_mod, "link");
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);

    SCFSolver<SchwartzPrescreening, Link> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &prescreening_alg, &link_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "prescreening") == 0 
           && strcmp(exchange_alg_str, "prescreening") == 0) {
    
    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    SCFSolver<SchwartzPrescreening, SchwartzPrescreening> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &prescreening_alg, &prescreening_alg, 
                num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "prescreening") == 0 
           && strcmp(exchange_alg_str, "multi") == 0) {
    
    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);

    SCFSolver<SchwartzPrescreening, MultiTreeFock> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &prescreening_alg, &multi_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "prescreening") == 0 
           && strcmp(exchange_alg_str, "naive") == 0) {
    
    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    SCFSolver<SchwartzPrescreening, NaiveFockMatrix> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &prescreening_alg, &naive_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "multi") == 0 
           && strcmp(exchange_alg_str, "link") == 0) {
    
    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);

    Link link_alg;
    fx_module* link_mod = fx_submodule(root_mod, "link");
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);

    SCFSolver<MultiTreeFock, Link> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &multi_alg, &link_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "multi") == 0 
           && strcmp(exchange_alg_str, "prescreening") == 0) {
    
    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);

    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    SCFSolver<MultiTreeFock, SchwartzPrescreening> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &multi_alg, &prescreening_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "multi") == 0 
           && strcmp(exchange_alg_str, "multi") == 0) {
    
    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);

    SCFSolver<MultiTreeFock, MultiTreeFock> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &multi_alg, &multi_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "multi") == 0 
           && strcmp(exchange_alg_str, "naive") == 0) {
    
    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);

    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    SCFSolver<MultiTreeFock, NaiveFockMatrix> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &multi_alg, &naive_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "naive") == 0 
           && strcmp(exchange_alg_str, "link") == 0) {
    
    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    Link link_alg;
    fx_module* link_mod = fx_submodule(root_mod, "link");
    link_alg.Init(centers, exp_mat, momenta, density, link_mod);

    SCFSolver<NaiveFockMatrix, Link> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &naive_alg, &link_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "naive") == 0 
           && strcmp(exchange_alg_str, "prescreening") == 0) {
    
    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    SchwartzPrescreening prescreening_alg;
    fx_module* prescreening_mod = fx_submodule(root_mod, "prescreening");
    prescreening_alg.Init(centers, exp_mat, momenta, density, prescreening_mod);
    
    SCFSolver<NaiveFockMatrix, SchwartzPrescreening> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &naive_alg, &prescreening_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else if (strcmp(coulomb_alg_str, "naive") == 0 
           && strcmp(exchange_alg_str, "multi") == 0) {
    
    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    MultiTreeFock multi_alg;
    fx_module* multi_mod = fx_submodule(root_mod, "multi");
    multi_alg.Init(centers, exp_mat, momenta, density, multi_mod);

    SCFSolver<NaiveFockMatrix, MultiTreeFock> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &naive_alg, &multi_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else*/ if (strcmp(coulomb_alg_str, "naive") == 0 
           && strcmp(exchange_alg_str, "naive") == 0) {
  
    NaiveFockMatrix naive_alg;
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);

    SCFSolver<NaiveFockMatrix, NaiveFockMatrix> solver;
    solver.Init(centers, exp_mat, momenta, density, solver_mod, nuclear_centers, 
                nuclear_charges, &naive_alg, &naive_alg, num_electrons);
    solver.ComputeWavefunction();
  }
  else {
    FATAL("Incorrectly specified fock matrix algorithms.\n");
  }
  
  eri::ERIFree();
  fx_done(root_mod);

  return 0;

} // main