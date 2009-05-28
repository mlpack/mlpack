#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "generative_mmk.h"
#include "utils.h"


void GetDnaData(int n_states,
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

void GetStrawmanData(ArrayList<HMM<Multinomial> > *p_hmms,
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

void TestHMMClassification(const ArrayList<HMM<Multinomial> > &hmms,
			   const GenVector<int> &labels) {
  /*
  ArrayList<HMM<Multinomial> > hmms;
  GenVector<int> labels;
  
  //GetStrawmanData(&hmms, &labels);
  GetDnaData(10, &hmms, &labels);
  */
  int n_hmms = labels.length();
  printf("n_hmms = %d\n", n_hmms);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  printf("lambda = %f\n", lambda);
  int witness_length = fx_param_int(NULL, "witness_length", 70);
  printf("witness_length = %d\n", witness_length);


  Matrix kernel_matrix;
  GenerativeMMKBatch(lambda, witness_length, hmms, &kernel_matrix);
  //PrintDebug("original kernel matrix", kernel_matrix, "%3e");

  Vector original_eigenvalues;
  la::EigenvaluesInit(kernel_matrix,
		      &original_eigenvalues);
  //PrintDebug("original eigenvalues", original_eigenvalues, "%3e");

  NormalizeKernelMatrix(&kernel_matrix);
  //PrintDebug("normalized kernel matrix", kernel_matrix, "%3e");

  Vector normalized_eigenvalues;
  la::EigenvaluesInit(kernel_matrix,
		      &normalized_eigenvalues);
  //PrintDebug("normalized eigenvalues", normalized_eigenvalues, "%3e");
  //data::Save("kernel_matrix.csv", kernel_matrix);


  
  Matrix id_label_pairs;
  id_label_pairs.Init(2, n_hmms);
  for(int i = 0; i < n_hmms; i++) {
    id_label_pairs.set(0, i, i);
    id_label_pairs.set(1, i, labels[i]);
  }

  id_label_pairs.PrintDebug("id_label_pairs");

  Vector c_set;
  int min_c_exp = -15;
  int max_c_exp = 4;
  c_set.Init(max_c_exp - min_c_exp + 1);
  for(int i = min_c_exp; i <= max_c_exp; i++) {
    c_set[i - min_c_exp] = pow(2, i);
  }
  PrintDebug("c_set", c_set, "%3e");

  
  SVMKFoldCV(id_label_pairs, kernel_matrix, c_set);

}

void SaveDNAHMMs() {
  const char* hmms_filename = "frozen_dna_hmms_topo2";
  const char* labels_filename = "frozen_dna_labels";
  
  struct stat stFileInfo;
  if(stat(hmms_filename, &stFileInfo) == 0) {
    FATAL("Error: File to which HMMs are to be saved already exists! Bypassing learning and exiting...");
  }

  ArrayList<HMM<Multinomial> > hmms;
  GenVector<int> labels;
  
  //GetStrawmanData(&hmms, &labels);
  GetDnaData(2, &hmms, &labels);
  
  WriteOutOTObject(hmms_filename, hmms);
  WriteOutOTObject(labels_filename, labels);
}


void LoadDNAHMMs(ArrayList<HMM<Multinomial> >* p_hmms,
		 GenVector<int>* p_labels) {
  ReadInOTObject("frozen_dna_hmms_topo2", p_hmms);
  ReadInOTObject("frozen_dna_labels", p_labels);

  
//   PrintDebug("labels", *p_labels, "%d");
//   char hmm_name[80];
//   for(int i = 0; i < 1000; i++) {
//     sprintf(hmm_name, "hmm %d", i);
//     (*p_hmms)[i].PrintDebug(hmm_name);
//   }
  
}




int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

//   SaveDNAHMMs();

  ArrayList<HMM<Multinomial> > hmms;
  GenVector<int> labels;
  LoadDNAHMMs(&hmms, &labels);
  TestHMMClassification(hmms, labels);
  
  fx_done(fx_root);
}
