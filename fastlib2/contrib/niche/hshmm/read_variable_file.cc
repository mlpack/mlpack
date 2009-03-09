#include "hmm_testing.h"

int main(int argc, char* argv[]) {

  ArrayList<Vector> data;

  LoadVaryingLengthData("/home/niche/Desktop/exons.dat", &data);

  data[data.size() - 1].PrintDebug("last sequence");

  printf("%d sequences\n", data.size());
  
}
