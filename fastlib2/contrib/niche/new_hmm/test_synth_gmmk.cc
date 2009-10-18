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



void GetSynthHMMs(int n_states,
		ArrayList<HMM<Multinomial> > *p_hmms,
		GenVector<int> *p_labels) {
  ArrayList<HMM<Multinomial> > &hmms = *p_hmms;
  GenVector<int> &labels = *p_labels;
  
  const char* class1_filename = "synth1000_pos.dat";
  const char* class0_filename = "synth1000_neg.dat";

  int n_dims = 2;

  bool compute_n_states = (n_states == -1);

  double ratio = fx_param_double(NULL, "ratio", 0.1);

  ArrayList<GenMatrix<int> > class1_sequences;
  LoadVaryingLengthData(class1_filename, &class1_sequences);
  int n_class1 = class1_sequences.size();
  
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


void GetStrawmanHMMs(ArrayList<HMM<Multinomial> > *p_hmms,
		  GenVector<int> *p_labels) {
  ArrayList<HMM<Multinomial> > &hmms = *p_hmms;
  GenVector<int> &labels = *p_labels;

  int n_hmms = 100;
  int half_n_hmms = n_hmms / 2;
  n_hmms = 2 * half_n_hmms;

  hmms.Init(n_hmms);
  labels.Init(n_hmms);

  ArrayList<ArrayList<GenMatrix<int> > > sequences;
  sequences.Init(n_hmms);

  for(int i = 0; i < n_hmms; i++) {
    sequences[i].Init(1);
    sequences[i][0].Init(1, 100);
  }

  for(int i = 0; i < half_n_hmms; i++) {
    for(int t = 0; t < 100; t++) {
      sequences[i][0].set(0, t, t >= 50);
    }
    labels[i] = 1;
  }
  
  for(int i = half_n_hmms; i < n_hmms; i++) {
    for(int t = 0; t < 100; t++) {
      sequences[i][0].set(0, t, t < 50);
    }
    labels[i] = 0;
  }
  
  for(int i = 0; i < n_hmms; i++) {
    printf("sequences[%d]\n", i);
    for(int t = 0; t < 100; t++) {
      printf("%d ", sequences[i][0].get(0, t));
    }
    printf("\n");
  }

  srand48(time(0));  
  for(int i = 0; i < n_hmms; i++) {
    hmms[i].Init(2, 2, MULTINOMIAL);

    hmms[i].InitParameters(sequences[i]);
  
    hmms[i].PrintDebug("hmm after calling InitParameters(sequences)");
    
    hmms[i].ViterbiUpdate(sequences[i]);
    
    hmms[i].PrintDebug("hmm after calling ViterbiUpdate(sequences)");
  
    hmms[i].BaumWelch(sequences[i],
		      1e-6 * ((double)1),
		      1000);
    
    hmms[i].PrintDebug("hmm after calling BaumWelch");
  }
}

void SaveSynthHMMs() {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* hmms_partial_filename = "frozen_synth_hmms_topo";
  char hmms_filename[80];
  if(n_states > 0) {
    sprintf(hmms_filename, "%s%d", hmms_partial_filename, n_states);
  }
  else if(n_states == -1) {
    sprintf(hmms_filename, "%sJ", hmms_partial_filename);
  }
  else {
    FATAL("Error: Invalid choice of n_states. Valid settings are integers greater than 0 to specify same number of states for every sequence's HMM OR -1 to specify formula-based number of states for each sequence's HMM. Exiting...");
  }
  printf("hmms_filename = \"%s\"\n", hmms_filename);
  const char* labels_filename = "frozen_synth_labels";
  
  struct stat stFileInfo;
  if(stat(hmms_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMMs are to be saved already exists! Bypassing learning and exiting...");
  }

  ArrayList<HMM<Multinomial> > hmms;
  GenVector<int> labels;
  
  GetSynthHMMs(n_states, &hmms, &labels);
  
  WriteOutOTObject(hmms_filename, hmms);
  WriteOutOTObject(labels_filename, labels);
}


void LoadSynthHMMs(ArrayList<HMM<Multinomial> >* p_hmms,
		   GenVector<int>* p_labels) {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* hmms_partial_filename = "../../../../frozen_synth_hmms_topo";
  char hmms_filename[80];
  if(n_states > 0) {
    sprintf(hmms_filename, "%s%d", hmms_partial_filename, n_states);
  }
  else if(n_states == -1) {
    sprintf(hmms_filename, "%sJ", hmms_partial_filename);
  }
  else {
    FATAL("Error: Invalid choice of n_states. Valid settings are integers greater than 0 to specify same number of states for every sequence's HMM OR -1 to specify formula-based number of states for each sequence's HMM. Exiting...");
  }
  //  sprintf(hmms_filename, "%s%d", hmms_partial_filename, n_states);
  printf("hmms_filename = \"%s\"\n", hmms_filename);
  const char* labels_filename = "../../../../frozen_synth_labels";

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
    SaveSynthHMMs();
  }
  else if(strcmp(mode, "test") == 0) {
    ArrayList<HMM<Multinomial> > hmms;
    GenVector<int> labels;
    LoadSynthHMMs(&hmms, &labels);
    TestHMMGenMMK2Classification(hmms, labels);
  }
  else {
    FATAL("Error: Invalid choice of parameter /mode. Valid settings are \"save\" and \"test\". Exiting...");
  }
  
  fx_done(fx_root);
}
