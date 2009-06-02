#include "fastlib/fastlib.h"
#include "test_engine.h"
#include "utils.h"


void LoadSequencesAndLabels(ArrayList<GenMatrix<int> >* p_sequences,
			    GenVector<int>* p_labels) {
  ArrayList<GenMatrix<int> > &sequences = *p_sequences;
  GenVector<int> &labels = *p_labels;

  const char* exons_filename = "../../../../exons_small.dat";
  const char* introns_filename = "../../../../introns_small.dat";

  //const char* exons_filename = "exons_small.dat";
  //const char* introns_filename = "introns_small.dat";


  LoadVaryingLengthData(exons_filename, &sequences);
  int n_exons = sequences.size();

  ArrayList<GenMatrix<int> > intron_sequences;
  LoadVaryingLengthData(introns_filename, &intron_sequences);
  int n_introns = intron_sequences.size();

  sequences.AppendSteal(&intron_sequences);

  int n_sequences = n_exons + n_introns;
  labels.Init(n_sequences);
  for(int i = 0; i < n_exons; i++) {
    labels[i] = 1;
  }
  for(int i = n_exons; i < n_sequences; i++) {
    labels[i] = 0;
  }
}


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  ArrayList<GenMatrix<int> > sequences;
  GenVector<int> labels;
  LoadSequencesAndLabels(&sequences, &labels);
  TestMarkovMMKClassification(4, sequences, labels);

  fx_done(fx_root);
}
