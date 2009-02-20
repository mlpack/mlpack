#include "fastlib/fastlib.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace std;

void LoadProfile(const char* name,
		 Matrix* p_transition, Matrix* all_p_emission) {
  FILE* file = fopen(name, "r");

  char* buffer = (char*) malloc(sizeof(char) * 100);
  size_t len = 0;

  double transition_nums[1000];
  double emission_nums[1000];

  int i_transition = 0;
  int i_emission = 0;
  
  getline(&buffer, &len, file);
  while(fscanf(file, "%lf,", transition_nums + i_transition) > 0) {
    i_transition++;
  }

  getline(&buffer, &len, file);
  while(fscanf(file, "%lf,", emission_nums + i_emission) > 0) {
    i_emission++;
  }
  fclose(file);

  //printf("i_transition = %d\n", i_transition);
  //printf("i_emission = %d\n", i_emission);


  int n_states = (int) sqrt((double) i_transition);
  int n_symbols = (int) (((double) i_emission) / ((double) n_states));

  //printf("n_states = %d\nn_symbols = %d\n", n_states, n_symbols);


  p_transition -> Init(n_states, n_states);
  int k = 0;
  for(int i = 0; i < n_states; i++) {
    for(int j = 0; j < n_states; j++) {
      p_transition -> set(i, j, transition_nums[k]);
      k++;
    }
  }

  //p_transition -> PrintDebug("p_transition");

  // in the profile file columns sum to 1, but in Matrix I need rows to sum to 1
  la::TransposeSquare(p_transition);

  all_p_emission -> Init(n_symbols, n_states);
  k = 0;
  for(int i = 0; i < n_symbols; i++) {
    for(int j = 0; j < n_states; j++) {
      all_p_emission -> set(i, j, emission_nums[k]);
      k++;
    }
  }
  /*
  //all_p.PrintDebug("all_p");

  for(int i = 0; i < n_states; i++) {
  Vector p;
  all_p.MakeColumnVector(i, &p);
  printf("state %d\n", i);
  p.PrintDebug("p");
  }
  */
}

/*
int main(int argc, char* argv[]) {
  LoadProfile("../../tqlong/mmf/profiles/est_bw_pro_000.dis");

  return 1;
}
*/
