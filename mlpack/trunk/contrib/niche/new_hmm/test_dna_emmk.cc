#include "fastlib/fastlib.h"
#include "test_dna_utils.h"
#include "test_engine.h"
#include "utils.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  ArrayList<GenMatrix<int> > sequences;
  GenVector<int> labels;
  LoadSequencesAndLabels(&sequences, &labels);
  TestMarkovMMKClassification(4, sequences, labels);

  fx_done(fx_root);
}
