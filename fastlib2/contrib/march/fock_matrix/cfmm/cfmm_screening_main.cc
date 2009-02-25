#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "contrib/dongryel/fast_multipole_method/continuous_fmm.h"
#include "contrib/march/fock_matrix/naive/naive_fock_matrix.h"

const fx_entry_doc cfmm_screening_entries[] = {
{"centers", FX_REQUIRED, FX_STR, NULL, 
 "A file containing the centers of the basis functions.\n"},
{"exponents", FX_REQUIRED, FX_STR, NULL, 
"A file containing the exponents of the basis functions.\n"
"Must have the same number of rows as centers.\n"},
{"density", FX_PARAM, FX_STR, NULL, 
"A file containing the density matrix.  If it is not provided, an all-ones\n"
"matrix is assumed.\n"},
  {"threshold", FX_PARAM, FX_DOUBLE, NULL,
  "The threshold for cutting off a shell-pair.  Default: 10e-10\n"},
  {"centers_out", FX_PARAM, FX_STR, NULL,
  "The file to write the charge centers to.  Default: centers.csv \n"},
  {"exponents_out", FX_PARAM, FX_STR, NULL, 
  "The file to write the charge exponents to.  Default: exp.csv\n"}
};

const fx_module_doc cfmm_screening_main_doc = {
  cfmm_screening_entries, NULL, 
  "Selects significant charge distributions for use in the CFMM.\n"
};

int main(int argc, char* argv[]) {

  //  fx_module* root_mod = fx_init(argc, argv, &cfmm_screening_main_doc);
  fx_module* root_mod = fx_init(argc, argv, &cfmm_screening_main_doc);

  Matrix centers;
  const char* centers_file = fx_param_str_req(root_mod, "centers");
  data::Load(centers_file, &centers);
  
  Matrix exp_mat;
  const char* exp_file = fx_param_str_req(root_mod, "exponents");
  data::Load(exp_file, &exp_mat);
  
  Matrix mom;
  mom.Init(1, centers.n_cols());
  mom.SetAll(0);
  
  
  if (centers.n_cols() != exp_mat.n_cols()) {
    FATAL("Number of basis centers must equal number of exponents.\n");
  }
  
  double thresh;
  thresh = fx_param_double(root_mod, "threshold", 10e-10);
  
  index_t num_shells = centers.n_cols();
  ArrayList<BasisShell> shells;
  shells.Init(num_shells);

  /*
  centers.PrintDebug();
  exp_mat.PrintDebug();
  printf("centers: cols: %d, rows %d\n", centers.n_cols(), centers.n_rows());
  printf("exp: cols: %d, rows %d\n", exp_mat.n_cols(), exp_mat.n_rows());
  */

  for (index_t i = 0; i < num_shells; i++) {
  
    Vector new_cent;
    centers.MakeColumnVector(i, &new_cent);
  
    shells[i].Init(new_cent, exp_mat.ref(0, i), 0, i);
  
  } // for i
  
  fx_result_int(root_mod, "num_basis_functions", num_shells);
  
  Matrix density_mat;
  if (fx_param_exists(root_mod, "density")) {
    const char* density_file = fx_param_str_req(root_mod, "density");
  }
  else {
    density_mat.Init(num_shells, num_shells);
    density_mat.SetAll(1.0);
  }
  
  
  ArrayList<ShellPair> shell_pairs;
  
  fx_timer_start(root_mod, "cfmm_time");
  
  fx_timer_start(root_mod, "prescreening_time");
  index_t num_shell_pairs = eri::ComputeShellPairs(&shell_pairs, shells, 
                                                   thresh);
                                                   
                
                                                   
  Matrix charge_centers;
  charge_centers.Init(3,num_shell_pairs);
  
  Matrix charge_exponents;
  charge_exponents.Init(1,num_shell_pairs);


  for (index_t i = 0; i < num_shell_pairs; i++) {
  
    Vector cent_vec;
    double new_exp = eri::ComputeGPTCenter(shell_pairs[i].M_Shell().center(), 
                                           shell_pairs[i].M_Shell().exp(), 
                                           shell_pairs[i].N_Shell().center(), 
                                           shell_pairs[i].N_Shell().exp(), 
                                           &cent_vec);
  
    // add to output matrix
    charge_centers.CopyVectorToColumn(i, cent_vec);
    
    charge_exponents.set(0,i, new_exp);
  
  } // for i
  
  
  // Replace this with reading in the matrix and processing it later
  Matrix densities;
  densities.Init(1,num_shell_pairs);
  
  for (index_t i = 0; i < num_shell_pairs; i++) {
  
    index_t m_ind = shell_pairs[i].M_index();
    index_t n_ind = shell_pairs[i].N_index();
    double new_density = density_mat.ref(m_ind, n_ind);
    
    // Lumping the coefficients of the integral into the charges
    new_density = new_density * shell_pairs[i].integral_factor();
    new_density = new_density / (pow(shell_pairs[i].exponent(), 1.5));
    //printf("exp: %g\n", shell_pairs[i].exponent());
    new_density = new_density * shell_pairs[i].M_Shell().normalization_constant();
    new_density = new_density * shell_pairs[i].N_Shell().normalization_constant();
    
    // If not on diagonal, then needs to be counted twice
    if (m_ind != n_ind) {
      new_density = 2 * new_density;
    }
  
    printf("m_ind: %d, n_ind: %d, new_density: %g\n", m_ind, n_ind, new_density);
  
    densities.set(0, i, new_density);
  
  } // for i
  
  
  fx_timer_stop(root_mod, "prescreening_time");
  
  
  /*const char* exp_out_file = fx_param_str(root_mod, "exponents_out", "exp.csv");
  data::Save(exp_out_file, charge_exponents);
  
  const char* centers_out_file = fx_param_str(root_mod, "centers_out", 
                                              "centers.csv");
  data::Save(centers_out_file, charge_centers);
  */
  
  struct datanode* cfmm_module = fx_submodule(root_mod, "cfmm");
  
  ContinuousFmm cfmm_algorithm;
  cfmm_algorithm.Init(charge_centers, charge_centers, densities, 
                      charge_exponents, true, cfmm_module);
  
  Vector cfmm_output;
  
  fx_timer_start(root_mod, "multipole_time");
  cfmm_algorithm.Compute(&cfmm_output);
  fx_timer_stop(root_mod, "multipole_time");
  
  DEBUG_ASSERT(cfmm_output.length() == num_shell_pairs);
  
  // Re-arrange the cfmm output into a matrix
  Matrix cfmm_fock;
  cfmm_fock.Init(num_shells, num_shells);
  
  for (index_t i = 0; i < num_shell_pairs; i++) {
  
    index_t m_ind = shell_pairs[i].M_index();
    index_t n_ind = shell_pairs[i].N_index();
    
    // multiply by prefactors and such
    
    double fock_entry = cfmm_output[i];

    printf("fock_entry (before factors): %g\n", fock_entry);

    fock_entry = fock_entry * shell_pairs[i].integral_factor();
    fock_entry = fock_entry / (pow(shell_pairs[i].exponent(), 1.5));
    fock_entry = fock_entry * pow(math::PI, 3.0);
    fock_entry = fock_entry * shell_pairs[i].M_Shell().normalization_constant();
    fock_entry = fock_entry * shell_pairs[i].N_Shell().normalization_constant();

    printf("fock_entry (after factors): %g\n", fock_entry);
    
    cfmm_fock.set(m_ind, n_ind, fock_entry);
    cfmm_fock.set(n_ind, m_ind, fock_entry);
    
  } // for i
  
  fx_timer_stop(root_mod, "cfmm_time");
  
  printf("CFMM Coulomb\n");
  cfmm_fock.PrintDebug();
  
  
  /////////////// Naive CFMM ///////////////////////////
  
  if (fx_param_exists(root_mod, "do_cfmm_naive")) {
    fx_timer_start(root_mod, "naive_cfmm_time");
    Vector naive_results;
    cfmm_algorithm.NaiveCompute(&naive_results);
    fx_timer_stop(root_mod, "naive_cfmm_time");
    
    Matrix naive_cfmm_fock;
    naive_cfmm_fock.Init(num_shells, num_shells);
    
    for (index_t i = 0; i < num_shell_pairs; i++) {
      
      index_t m_ind = shell_pairs[i].M_index();
      index_t n_ind = shell_pairs[i].N_index();
      
      // multiply by prefactors and such
      
      double fock_entry = naive_results[i];
      
      //printf("fock_entry (before factors): %g\n", fock_entry);
      
      fock_entry = fock_entry * shell_pairs[i].integral_factor();
      fock_entry = fock_entry / (pow(shell_pairs[i].exponent(), 1.5));
      fock_entry = fock_entry * pow(math::PI, 3.0);
      fock_entry = fock_entry * shell_pairs[i].M_Shell().normalization_constant();
      fock_entry = fock_entry * shell_pairs[i].N_Shell().normalization_constant();
      
      //printf("fock_entry (after factors): %g\n", fock_entry);
      
      naive_cfmm_fock.set(m_ind, n_ind, fock_entry);
      naive_cfmm_fock.set(n_ind, m_ind, fock_entry);
      
    } // for i
    
    printf("Naive CFMM\n");
    naive_cfmm_fock.PrintDebug();
    
    
  }
  
  
  
  
  NaiveFockMatrix naive_comp;
  
  if (fx_param_exists(root_mod, "do_naive")) {
    fx_module* naive_mod = fx_submodule(root_mod, "naive");
    naive_comp.Init(centers, exp_mat, mom, density_mat, naive_mod);
    naive_comp.ComputeFock();
    Matrix naive_fock;
    naive_comp.OutputFock(NULL, &naive_fock, NULL);
    
    printf("naive_fock\n");
    naive_fock.PrintDebug();
    
  }
  
  
  fx_result_int(root_mod, "num_charge_dists", num_shell_pairs);
  
  fx_done(root_mod);
  
  return 0;

} // main()
