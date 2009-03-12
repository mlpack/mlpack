#include "hmm_testing.h"

int main(int argc, char* argv[]) {

  ArrayList<Vector> data;

  LoadVaryingLengthData("introns_small.dat", &data);

  int n_sequences = data.size();
  Matrix lengths;
  lengths.Init(n_sequences, 1);

  int max_length = 0;
  int sum_length = 0;
  for(int i = 0; i < n_sequences; i++) {
    lengths.set(i, 0, data[i].length());
    sum_length += data[i].length();
    if(data[i].length() > max_length) {
      max_length = data[i].length();
    }
  }

  data::Save("lengths.dat", lengths);


  printf("max_length = %d\n", max_length);

  printf("mean length = %f\n", ((double)sum_length) / ((double)n_sequences));


  printf("n_sequences = %d\n", n_sequences);

  printf("sum_length = %d\n", sum_length);


  //data[data.size() - 1].PrintDebug("last sequence");

  printf("%d sequences\n", data.size());
  
}
