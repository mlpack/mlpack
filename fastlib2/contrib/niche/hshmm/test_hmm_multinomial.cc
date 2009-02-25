#include "fastlib/fastlib.h"
#include "hmm.h"
#include "multinomial.h"
#include "mmk.h"
#include "load_profile.cc"

const fx_entry_doc hshmm_main_entries[] = {
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc hshmm_main_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc hshmm_main_doc = {
  hshmm_main_entries, hshmm_main_submodules,
  "This is used for a Hilbert-Schmidt embedding of an HMM.\n"
};



double ComputeKernel(const char* filename_profile_hmm_1,
		     const char* filename_profile_hmm_2) {

  int n_dims = 2;
  int T = 70;

  double lambda = fx_param_double_req(NULL, "lambda");

  Matrix p_transition;
  Matrix all_p_emission;

  LoadProfile(filename_profile_hmm_1, &p_transition, &all_p_emission);
  int n_states_a = p_transition.n_cols();
  HMM<Multinomial> hmm_a;
  hmm_a.Init(n_states_a, n_dims, T);
  hmm_a.RandomlyInitialize();
  hmm_a.SetPTransition(p_transition);

  for(int i = 0; i < n_states_a; i++) {
    Vector p;
    all_p_emission.MakeColumnVector(i, &p);

    Multinomial multinomial;
    multinomial.Init(p.length());
    multinomial.SetP(p);

    hmm_a.SetStateDistribution(i, multinomial);
  }

  hmm_a.ComputeCumulativePTransition();
  hmm_a.ComputeStateProbabilities();
  /*
  hmm_a.cumulative_p_transition().PrintDebug("a cumulative_p_transition");
  hmm_a.state_probabilities().PrintDebug("a state_probabilities");
  hmm_a.state_cumulative_probabilities().PrintDebug("a state_cumulative_probabilities");
  */
  
  p_transition.Destruct();
  all_p_emission.Destruct();

  LoadProfile(filename_profile_hmm_2, &p_transition, &all_p_emission);
  int n_states_b = p_transition.n_cols();
  HMM<Multinomial> hmm_b;
  hmm_b.Init(n_states_b, n_dims, T);
  hmm_b.RandomlyInitialize();
  hmm_b.SetPTransition(p_transition);

  for(int i = 0; i < n_states_b; i++) {
    Vector p;
    all_p_emission.MakeColumnVector(i, &p);

    Multinomial multinomial;
    multinomial.Init(p.length());
    multinomial.SetP(p);

    hmm_b.SetStateDistribution(i, multinomial);
  }

  hmm_b.ComputeCumulativePTransition();
  hmm_b.ComputeStateProbabilities();
  /*
  hmm_b.cumulative_p_transition().PrintDebug("b cumulative_p_transition");
  hmm_b.state_probabilities().PrintDebug("b state_probabilities");
  hmm_b.state_cumulative_probabilities().PrintDebug("b state_cumulative_probabilities");
  */
  /*
  hmm_a.PrintDebug("a");
  hmm_b.PrintDebug("b");
  */
  //printf("Ready to MMK\n");
  
  MeanMapKernel mmk;
  mmk.Init(lambda, T);

  double val = mmk.Compute(hmm_a, hmm_b);

  printf("hmm_dist(hmm_a, hmm_b) = %e\n",
	 val);

  return val;
}



int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, &hshmm_main_doc);

  int n_hmms = 100;

  Matrix kernel_matrix;
  kernel_matrix.Init(n_hmms, n_hmms);
  kernel_matrix.SetAll(-1);

  for(int i_hmm_1 = 0; i_hmm_1 < n_hmms; i_hmm_1++) {
    char filename_profile_hmm_1[100];
    sprintf(filename_profile_hmm_1, "../../tqlong/mmf/profiles/est_mmf_pro_%03d.dis", i_hmm_1);
    for(int i_hmm_2 = i_hmm_1; i_hmm_2 < n_hmms; i_hmm_2++) {
      char filename_profile_hmm_2[100];
      sprintf(filename_profile_hmm_2, "../../tqlong/mmf/profiles/est_mmf_pro_%03d.dis", i_hmm_2);

      double val = ComputeKernel(filename_profile_hmm_1, filename_profile_hmm_2);
      kernel_matrix.set(i_hmm_1, i_hmm_2, val);
      if(i_hmm_1 != i_hmm_2) {
	kernel_matrix.set(i_hmm_2, i_hmm_1, val);
      }
    }
  }

  data::Save("kernel_mat.dat", kernel_matrix);

  fx_done(root);

  return SUCCESS_PASS;
}


  

  


