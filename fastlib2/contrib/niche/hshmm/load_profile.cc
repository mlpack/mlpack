#include "iostream"

using namespace std;

void LoadProfile(const char* name) {
  FILE* file = fopen(name, "r");

  char* buffer = (char*) malloc(sizeof(char) * 100);
  size_t len = 0;
  size_t read;

  double transition_nums[1000];
  double emission_nums[1000];

  int i_transition = 0;
  int i_emission = 0;
  
  int state = 0;
  getline(&buffer, &len, file);
  while(fscanf(file, "%f,", transition_nums + i_transition) > 0) {
    i_transition++;
  }

  getline(&buffer, &len, file);
  while(fscanf(file, "%f,", emission_nums + i_emission) > 0) {
    i_emission++;
  }
  fclose(file);

  printf("i_transition = %d\n", i_transition);
  printf("i_emission = %d\n", i_emission);




}


int main(int argc, char* argv[]) {
  LoadProfile("/scratch/niche/fastlib2/contrib/tqlong/mmf/profiles/est_bw_pro_000.dis");

  return 1;
}
