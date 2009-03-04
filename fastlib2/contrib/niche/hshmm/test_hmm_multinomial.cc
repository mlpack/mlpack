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



double ComputeKernel(const Matrix &transition_matrix_a,
		     const Matrix &emission_matrix_a,
		     const Matrix &transition_matrix_b,
		     const Matrix &emission_matrix_b) {
  
  int n_dims = emission_matrix_a.n_cols();
  int T = fx_param_int(NULL, "T", 70);

  double lambda = fx_param_double(NULL, "lambda", 0.3);

  int n_states_a = transition_matrix_a.n_cols();
  HMM<Multinomial> hmm_a;
  hmm_a.Init(n_states_a, n_dims, T);
  hmm_a.RandomlyInitialize();
  hmm_a.SetPTransition(transition_matrix_a);

  Matrix emission_matrix_a_transpose;
  la::TransposeInit(emission_matrix_a, &emission_matrix_a_transpose);
  for(int i = 0; i < n_states_a; i++) {
    Vector p;
    emission_matrix_a_transpose.MakeColumnVector(i, &p);

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

  int n_states_b = transition_matrix_b.n_cols();
  HMM<Multinomial> hmm_b;
  hmm_b.Init(n_states_b, n_dims, T);
  hmm_b.RandomlyInitialize();
  hmm_b.SetPTransition(transition_matrix_b);

  Matrix emission_matrix_b_transpose;
  la::TransposeInit(emission_matrix_b, &emission_matrix_b_transpose);
  for(int i = 0; i < n_states_b; i++) {
    Vector p;
    emission_matrix_b_transpose.MakeColumnVector(i, &p);

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
  
  //printf("hmm_dist(hmm_a, hmm_b) = %e\n", val);
  
  return val;
}


void ComputeKernelAllPairs(int n_sequences,
			   const ArrayList<Vector> &initial_probs_vectors,
			   const ArrayList<Matrix> &transition_matrices,
			   const ArrayList<Matrix> &emission_matrices,
			   Matrix *kernel_matrix) {

  HMM<Multinomial> hmms[n_sequences];
  
  int n_dims = emission_matrices[0].n_cols();
  int T = fx_param_int(NULL, "T", 70);

  double lambda = fx_param_double(NULL, "lambda", 0.3);
  
  for(int i = 0; i < n_sequences; i++) {
    int n_states = transition_matrices[i].n_cols();
    hmms[i].Init(n_states, n_dims, T);
    hmms[i].SetPInitial(initial_probs_vectors[i]);
    //hmms[i].RandomlyInitialize();
    hmms[i].SetPTransition(transition_matrices[i]);
    
    Matrix emission_matrix_transpose;
    la::TransposeInit(emission_matrices[i], &emission_matrix_transpose);
    for(int j = 0; j < n_states; j++) {
      Vector p;
      emission_matrix_transpose.MakeColumnVector(j, &p);
      Multinomial multinomial;
      multinomial.Init(p.length());
      multinomial.SetP(p);
      hmms[i].SetStateDistribution(j, multinomial);
    }
    hmms[i].ComputeCumulativePTransition();
    hmms[i].ComputeStateProbabilities();
    //hmms[i].PrintDebug("hmm");
  }
  
  
  kernel_matrix -> Init(n_sequences, n_sequences);
  
  
  MeanMapKernel mmk;
  mmk.Init(lambda, T);
  
  for(int i = 0; i < n_sequences; i++) {
    for(int j = 0; j < n_sequences; j++) {

      double val = mmk.Compute(hmms[j], hmms[i]);
      kernel_matrix -> set(j, i, val);
      if(i != j) {
	kernel_matrix -> set(i, j, val);
      }
    }
  }
  
}




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
  
  //printf("hmm_dist(hmm_a, hmm_b) = %e\n", val);
  
  return val;
}


/*
int main(int argc, char *argv[]) {
  fx_module* root = fx_init(argc, argv, &hshmm_main_doc);

  int n_hmms = 200;

  Matrix kernel_matrix;
  kernel_matrix.Init(n_hmms, n_hmms);
  kernel_matrix.SetAll(-1);

  for(int i = 0; i < n_hmms; i++) {
    printf("%d/%d\n", i, n_hmms);
    char filename_profile_hmm_1[100];
    sprintf(filename_profile_hmm_1, "profiles/est_mmf_pro_%03d.dis", i);
    for(int j = i; j < n_hmms; j++) {
      char filename_profile_hmm_2[100];
      sprintf(filename_profile_hmm_2, "profiles/est_mmf_pro_%03d.dis", j);

      double val = ComputeKernel(filename_profile_hmm_1, filename_profile_hmm_2);
      kernel_matrix.set(i, j, val);
      if(i != j) {
	kernel_matrix.set(j, i, val);
      }
    }
  }

  Vector sqrt_diag;
  sqrt_diag.Init(n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    sqrt_diag[i] = sqrt(kernel_matrix.get(i, i));
  }

  for(int i = 0; i < n_hmms; i++) {
    for(int j = 0; j < n_hmms; j++) {
      kernel_matrix.set(j, i,
			kernel_matrix.get(j, i) /
			(sqrt_diag[i] * sqrt_diag[j]));
    }
  }
			

  data::Save("kernel_mat.dat", kernel_matrix);

  fx_done(root);

  return SUCCESS_PASS;
}
*/

