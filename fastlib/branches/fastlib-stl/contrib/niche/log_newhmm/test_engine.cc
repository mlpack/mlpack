#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "utils.h"


void GetDnaHMMs(int n_states,
		ArrayList<HMM<Multinomial> > *p_hmms,
		GenVector<int> *p_labels) {
  ArrayList<HMM<Multinomial> > &hmms = *p_hmms;
  GenVector<int> &labels = *p_labels;
  
  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";

  int n_dims = 4;
  
  ArrayList<GenMatrix<int> > exon_sequences;
  LoadVaryingLengthData(exons_filename, &exon_sequences);
  int n_exons = exon_sequences.size();

  hmms.Init(n_exons);
  for(int i = 0; i < n_exons; i++) {
    printf("training on exon %d of %d\n", i, n_exons);
    ArrayList<GenMatrix<int> > one_sequence;
    one_sequence.Init(1);
    one_sequence[0].Init(0,0);
    one_sequence[0] = exon_sequences[i];

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

  
  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();

  ArrayList<HMM<Multinomial> > intron_hmms;
  intron_hmms.Init(n_introns);
  for(int i = 0; i < n_introns; i++) {
    printf("training on intron %d of %d\n", i, n_introns);
    ArrayList<GenMatrix<int> > one_sequence;
    one_sequence.Init(1);
    one_sequence[0].Init(0,0);
    one_sequence[0] = intron_sequences[i];

    intron_hmms[i].Init(n_states, n_dims, MULTINOMIAL);
    intron_hmms[i].InitParameters(one_sequence);
    //intron_hmms[i].PrintDebug("hmm after calling InitParameters(sequences)");
    intron_hmms[i].ViterbiUpdate(one_sequence);
    //intron_hmms[i].PrintDebug("hmm after calling ViterbiUpdate(sequences)");
    intron_hmms[i].BaumWelch(one_sequence,
		      1e-6 * ((double)1),
		      1000);
    //intron_hmms[i].PrintDebug("hmm after calling BaumWelch");
  }

  hmms.AppendSteal(&intron_hmms);
  
  int n_sequences = n_exons + n_introns;
  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = 1;
  }
  for(int i = n_exons; i < n_sequences; i++) {
    labels[i] = 0;
  }
}


void GetOneDnaHMM(int n_states,
		  HMM<Multinomial> *p_hmm,
		  GenVector<int> *p_labels) {
  HMM<Multinomial> &hmm = *p_hmm;
  GenVector<int> &labels = *p_labels;
  
  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";

  int n_dims = 4;
  
  ArrayList<GenMatrix<int> > sequences;
  LoadVaryingLengthData(exons_filename, &sequences);
  int n_exons = sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();


  sequences.AppendSteal(&intron_sequences);

  hmm.Init(n_states, n_dims, MULTINOMIAL);
  hmm.InitParameters(sequences);
  //hmm.PrintDebug("hmm after calling InitParameters(sequences)");
  hmm.ViterbiUpdate(sequences);
  //hmm.PrintDebug("hmm after calling ViterbiUpdate(sequences)");
  printf("%d sequences\n", sequences.size());
  hmm.BaumWelch(sequences,
		1e-6 * ((double)1),
		1000);
  //hmm.PrintDebug("hmm after calling BaumWelch");
  
  int n_sequences = n_exons + n_introns;
  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = 1;
  }
  for(int i = n_exons; i < n_sequences; i++) {
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



void SaveDNAHMMs() {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* hmms_partial_filename = "frozen_dna_hmms_topo";
  char hmms_filename[80];
  sprintf(hmms_filename, "%s%d", hmms_partial_filename, n_states);
  printf("hmms_filename = \"%s\"\n", hmms_filename);
  const char* labels_filename = "frozen_dna_labels";
  
  struct stat stFileInfo;
  if(stat(hmms_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMMs are to be saved already exists! Bypassing learning and exiting...");
  }

  ArrayList<HMM<Multinomial> > hmms;
  GenVector<int> labels;
  
  GetDnaHMMs(n_states, &hmms, &labels);
  
  WriteOutOTObject(hmms_filename, hmms);
  WriteOutOTObject(labels_filename, labels);
}


void LoadDNAHMMs(ArrayList<HMM<Multinomial> >* p_hmms,
		 GenVector<int>* p_labels) {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* hmms_partial_filename = "../../../../frozen_dna_hmms_topo";
  char hmms_filename[80];
  sprintf(hmms_filename, "%s%d", hmms_partial_filename, n_states);
  printf("hmms_filename = \"%s\"\n", hmms_filename);
  const char* labels_filename = "../../../../frozen_dna_labels";

  ReadInOTObject(hmms_filename, p_hmms);
  ReadInOTObject(labels_filename, p_labels);
  
//   PrintDebug("labels", *p_labels, "%d");
//   char hmm_name[80];
//   for(int i = 0; i < 1000; i++) {
//     sprintf(hmm_name, "hmm %d", i);
//     (*p_hmms)[i].PrintDebug(hmm_name);
//   }
  
}

void SaveOneDNAHMM() {
  int n_states = fx_param_int_req(NULL, "n_states");
  const char* one_hmm_partial_filename = "frozen_dna_one_hmm_topo";
  char one_hmm_filename[80];
  sprintf(one_hmm_filename, "%s%d", one_hmm_partial_filename, n_states);
  printf("one_hmm_filename = \"%s\"\n", one_hmm_filename);
  const char* labels_filename = "frozen_dna_labels";
  
  struct stat stFileInfo;
  if(stat(one_hmm_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMM is to be saved already exists! Bypassing learning and exiting...");
  }

  HMM<Multinomial> hmm;
  GenVector<int> labels;
  
  GetOneDnaHMM(n_states, &hmm, &labels);
  
  WriteOutOTObject(one_hmm_filename, hmm);
  WriteOutOTObject(labels_filename, labels);
}

void LoadOneDNAHMMAndSequences(HMM<Multinomial>* p_hmm,
			       ArrayList<GenMatrix<int> >* p_sequences,
			       GenVector<int>* p_labels) {
  ArrayList<GenMatrix<int> > &sequences = *p_sequences;

  int n_states = fx_param_int_req(NULL, "n_states");
  const char* one_hmm_partial_filename = "frozen_dna_one_hmm_topo";
  char one_hmm_filename[80];
  sprintf(one_hmm_filename, "%s%d", one_hmm_partial_filename, n_states);
  printf("one_hmm_filename = \"%s\"\n", one_hmm_filename);
  const char* labels_filename = "frozen_dna_labels";

  ReadInOTObject(one_hmm_filename, p_hmm);
  ReadInOTObject(labels_filename, p_labels);
  
//   PrintDebug("labels", *p_labels, "%d");
//   char hmm_name[80];
//   for(int i = 0; i < 1000; i++) {
//     sprintf(hmm_name, "hmm %d", i);
//     (*p_hmms)[i].PrintDebug(hmm_name);
//   }


  const char* exons_filename = "exons_small.dat";
  const char* introns_filename = "introns_small.dat";

  LoadVaryingLengthData(exons_filename, &sequences);

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);

  sequences.AppendSteal(&intron_sequences);
}



int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  ///// For Generative MMK /////

  const char* mode = fx_param_str_req(NULL, "mode");

  if(strcmp(mode, "save") == 0) {
    SaveDNAHMMs();
  }
  else if(strcmp(mode, "test") == 0) {
    ArrayList<HMM<Multinomial> > hmms;
    GenVector<int> labels;
    LoadDNAHMMs(&hmms, &labels);
    TestHMMGenMMKClassification(hmms, labels);
  }
  else {
    FATAL("Error: Invalid choice of parameter /mode. Valid settings are \"save\" and \"test\". Exiting...");
  }





  ///// For Latent MMK /////
  
  //SaveOneDNAHMM();
  
//   HMM<Multinomial> hmm;
//   ArrayList<GenMatrix<int> > sequences;
//   GenVector<int> labels;
//   LoadOneDNAHMMAndSequences(&hmm, &sequences, &labels);
//   TestHMMLatMMKClassification(hmm, sequences, labels);


  
  fx_done(fx_root);
}
