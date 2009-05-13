#include "zhmm.h"
#include "multinomial.h"

int main(int argc, char* argv[]) {
  printf("hi how are you\n");

  HMM<Multinomial> hmm;
  hmm.Init(3, 2);

  ArrayList<Matrix> sequences;
  sequences.Init(1);
  sequences[0].Init(100,1);
  for(int i = 0; i < 100; i++) {
    sequences[0].set(i, 0, i > 50);
  }

  hmm.BaumWelch(sequences);
}
