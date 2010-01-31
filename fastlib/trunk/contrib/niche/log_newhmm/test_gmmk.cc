#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "utils.h"

inline int ComputeNStates(double n_symbols,
			  double ratio,
			  double sequence_length) {
  return 
    ((int)floor(0.5 * sqrt(n_symbols * n_symbols
			   + 4.0 * (sequence_length * ratio + n_symbols + 1.0))
		- 0.5 * n_symbols))
    + 1;
}


void GetHMMs(int n_states,
		ArrayList<HMM<Multinomial> > *p_hmms,
		GenVector<int> *p_labels) {
  ArrayList<HMM<Multinomial> > &hmms = *p_hmms;
  GenVector<int> &labels = *p_labels;
  
  const char* class1_filename = 
    fx_param_str_req(NULL, "data_pos");
  const char* class0_filename = 
    fx_param_str_req(NULL, "data_neg");

  bool compute_n_states = (n_states == -1);

  double ratio = fx_param_double(NULL, "ratio", 0.1);

  ArrayList<GenMatrix<int> > class1_sequences;
  LoadVaryingLengthData(class1_filename, &class1_sequences);
  int n_class1 = class1_sequences.size();

  int n_dims = fx_param_int_req(NULL, "n_symbols");
  
  hmms.Init(n_class1);
  for(int i = 0; i < n_class1; i++) {
    printf("training on class1 %d of %d\n", i, n_class1);
    ArrayList<GenMatrix<int> > one_sequence;
    one_sequence.Init(1);
    one_sequence[0].Init(0,0);
    one_sequence[0] = class1_sequences[i];

    if(compute_n_states) {
      n_states = ComputeNStates(n_dims, ratio, class1_sequences[i].n_cols());
    }

    hmms[i].Init(n_states, n_dims, MULTINOMIAL);
    hmms[i].InitParameters(one_sequence);
    //hmms[i].PrintDebug("hmm after calling InitParameters(sequences)");
    hmms[i].ViterbiUpdate(one_sequence);
    //hmms[i].PrintDebug("hmm after calling ViterbiUpdate(sequences)");
    hmms[i].BaumWelch(one_sequence,
		      1e-6 * ((double)1),
		      1000);
    //hmms[i].PrintDebug("hmm after calling BaumWelch");
    //one_sequence.ReleasePtr();
  }

  
  ArrayList<GenMatrix<int> > class0_sequences;
  LoadVaryingLengthData(class0_filename, &class0_sequences);
  int n_class0 = class0_sequences.size();

  ArrayList<HMM<Multinomial> > class0_hmms;
  class0_hmms.Init(n_class0);
  for(int i = 0; i < n_class0; i++) {
    printf("training on class0 %d of %d\n", i, n_class0);
    ArrayList<GenMatrix<int> > one_sequence;
    one_sequence.Init(1);
    one_sequence[0].Init(0,0);
    one_sequence[0] = class0_sequences[i];

    if(compute_n_states) {
      n_states = ComputeNStates(n_dims, ratio, class0_sequences[i].n_cols());
    }

    class0_hmms[i].Init(n_states, n_dims, MULTINOMIAL);
    class0_hmms[i].InitParameters(one_sequence);
    //class0_hmms[i].PrintDebug("hmm after calling InitParameters(sequences)");
    class0_hmms[i].ViterbiUpdate(one_sequence);
    //class0_hmms[i].PrintDebug("hmm after calling ViterbiUpdate(sequences)");
    class0_hmms[i].BaumWelch(one_sequence,
		      1e-6 * ((double)1),
		      1000);
    //class0_hmms[i].PrintDebug("hmm after calling BaumWelch");
  }

  hmms.AppendSteal(&class0_hmms);
  
  int n_sequences = n_class1 + n_class0;
  labels.Init(n_sequences);
  for(int i = 0; i < n_class1; i++) {
    labels[i] = 1;
  }
  for(int i = n_class1; i < n_sequences; i++) {
    labels[i] = 0;
  }
}


void SaveHMMs() {
  int n_states = fx_param_int_req(NULL, "n_states");

  const char* experiment_name = fx_param_str_req(NULL, "exp");

  char hmms_filename[80];
  if(n_states > 0) {
    sprintf(hmms_filename, "frozen_%s_hmms_topo%d", experiment_name, n_states);
  }
  else if(n_states == -1) {
    sprintf(hmms_filename, "frozen_%s_hmms_topoJ", experiment_name);
  }
  else {
    FATAL("Error: Invalid choice of n_states. Valid settings are integers greater than 0 to specify same number of states for every sequence's HMM OR -1 to specify formula-based number of states for each sequence's HMM. Exiting...");
  }
  printf("hmms_filename = \"%s\"\n", hmms_filename);
  char labels_filename[80];
  sprintf(labels_filename, "frozen_%s_labels", experiment_name);
  
  struct stat stFileInfo;
  if(stat(hmms_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMMs are to be saved already exists! Bypassing learning and exiting...");
  }
  else if(stat(labels_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which labels are to be saved already exists! Bypassing learning and exiting...");
  }
  
  ArrayList<HMM<Multinomial> > hmms;
  GenVector<int> labels;
  
  GetHMMs(n_states, &hmms, &labels);
  
  WriteOutOTObject(hmms_filename, hmms);
  WriteOutOTObject(labels_filename, labels);
}


void LoadHMMs(ArrayList<HMM<Multinomial> >* p_hmms,
		   GenVector<int>* p_labels) {
  int n_states = fx_param_int_req(NULL, "n_states");

  const char* experiment_name =
    fx_param_str_req(NULL, "exp");

  char hmms_filename[80];
  if(n_states > 0) {
    sprintf(hmms_filename, "../../../../frozen_%s_hmms_topo%d", experiment_name, n_states);
  }
  else if(n_states == -1) {
    sprintf(hmms_filename, "../../../../frozen_%s_hmms_topoJ", experiment_name);
  }
  else {
    FATAL("Error: Invalid choice of n_states. Valid settings are integers greater than 0 to specify same number of states for every sequence's HMM OR -1 to specify formula-based number of states for each sequence's HMM. Exiting...");
  }
  //  sprintf(hmms_filename, "%s%d", hmms_partial_filename, n_states);
  printf("hmms_filename = \"%s\"\n", hmms_filename);
  char labels_filename[80];
  sprintf(labels_filename, "../../../../frozen_%s_labels", experiment_name);

  ReadInOTObject(hmms_filename, p_hmms);
  ReadInOTObject(labels_filename, p_labels);
  
//   PrintDebug("labels", *p_labels, "%d");
//   char hmm_name[80];
//   for(int i = 0; i < 1000; i++) {
//     sprintf(hmm_name, "hmm %d", i);
//     (*p_hmms)[i].PrintDebug(hmm_name);
//   }
  
}


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  const char* mode = fx_param_str_req(NULL, "mode");
  if(strcmp(mode, "save") == 0) {
    SaveHMMs();
  }
  else if(strcmp(mode, "test") == 0) {
    ArrayList<HMM<Multinomial> > hmms;
    GenVector<int> labels;
    LoadHMMs(&hmms, &labels);
    TestHMMGenMMK2Classification(hmms, labels);
  }
  else {
    FATAL("Error: Invalid choice of parameter /mode. Valid settings are \"save\" and \"test\". Exiting...");
  }
  
  fx_done(fx_root);
}
