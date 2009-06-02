#include "fastlib/fastlib.h"
#include "utils.h"


int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  const char* sampling_file_list = "inbio_data/sampling_file_list.txt";

  ArrayList<Matrix> samplings;
  samplings.Init(0);

  FILE* file = fopen(sampling_file_list, "r");
  
  char filename[80];
 
  int n_samplings = 0;
  while(fgets(filename, 80, file) != NULL) {
    filename[strlen(filename) - 1] = '\0'; // kill the newline character

    char full_filename[80];
    sprintf(full_filename, "inbio_data/%s", filename);
    
    samplings.PushBack(1);
    data::Load(full_filename, &(samplings[n_samplings]));
    n_samplings++;
  }

  fclose(file);

  fx_done(fx_root);
}
  
