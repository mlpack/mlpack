#include "fastlib/fastlib.h"
#include "contrib/tqlong/mmf/discreteHMM.h"
#include "contrib/tqlong/mmf/mmf3.cc"


void GenerateAndTrainSequences(const char* transition_filename,
			       const char* emission_filename,
			       int n_sequences,
			       Matrix transition_matrices[],
			       Matrix emission_matrices[]) {


  Matrix transition;
  data::Load(transition_filename, &transition);

  Matrix emission;
  data::Load(emission_filename, &emission);

  int n_symbols = emission.n_cols();


  DiscreteHMM hmm_gen;
  hmm_gen.Init(transition, emission);
  
  const int sequence_length = fx_param_int(NULL, "length", 100);

  int n_states = fx_param_int(NULL, "n_states", 3);
  
  int max_iter_mmf = fx_param_int(NULL, "max_iter_mmf", 500);
  int max_rand_iter = fx_param_int(NULL, "max_rand_iter", 60);
  double tol_mmf = fx_param_double(NULL, "tolerance_mmf", 1e-4);

  int max_iter_bw = fx_param_int(NULL, "max_iter_bw", 500);
  double tol_bw = fx_param_double(NULL, "tolerance_bw", 1e-3);



  for(int i = 0; i < n_sequences; i++) {
    Vector observed_sequence;
    Vector state_sequence;
    
    hmm_gen.GenerateSequence(sequence_length, &observed_sequence, &state_sequence);
    
    DiscreteHMM learned_hmm;
    learned_hmm.Init(n_states, n_symbols);
    
    ArrayList<Vector> seqs;
    seqs.Init(1);
    seqs[0].Copy(observed_sequence);
    
    
    learned_hmm.TrainMMF(seqs, max_rand_iter, max_iter_mmf, tol_mmf);
    
    learned_hmm.TrainBaumWelch(seqs, max_iter_bw, tol_bw);
    
    transition_matrices[i].Copy(learned_hmm.transmission());
    emission_matrices[i].Copy(learned_hmm.emission());
  }
  

}



int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  srand(time(0));
  
  int n_sequences = 100;

  Matrix class_0_transition_matrices[n_sequences];
  Matrix class_0_emission_matrices[n_sequences];
  GenerateAndTrainSequences("class_0_transition.csv", "class_0_emission.csv",
			    n_sequences,
			    class_0_transition_matrices,
			    class_0_emission_matrices);

  Matrix class_1_transition_matrices[n_sequences];
  Matrix class_1_emission_matrices[n_sequences];
  GenerateAndTrainSequences("class_1_transition.csv", "class_1_emission.csv",
			    n_sequences,
			    class_1_transition_matrices,
			    class_1_emission_matrices);
  
  fx_done(NULL);

  return SUCCESS_PASS;
}
