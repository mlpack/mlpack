#include "fastlib/fastlib.h"
#include "generative_mmk.h"
#include "kernel_pca.h"
#include "utils.h"


void LoadInbioData(ArrayList<Matrix>* p_samplings) {
  ArrayList<Matrix> &samplings = *p_samplings;

  const char* sampling_file_list = "inbio_data/sampling_file_list.txt";

  FILE* file = fopen(sampling_file_list, "r");
  
  char filename[80];
 
  int n_samplings = 0;
  samplings.Init(0);
  while(fgets(filename, 80, file) != NULL) {
    filename[strlen(filename) - 1] = '\0'; // kill the newline character

    char full_filename[80];
    sprintf(full_filename, "inbio_data/%s", filename);
    
    samplings.PushBack(1);
    data::Load(full_filename, &(samplings[n_samplings]));
    n_samplings++;
  }
  fclose(file);
}
  
int main(int argc, char* argv[]) {
  fx_init(argc, argv, NULL);

  ArrayList<Matrix> samplings;
  LoadInbioData(&samplings);
  ScaleSamplingsToCube(&samplings);


  
  Matrix kernel_matrix;
  KDEGenerativeMMKBatch(1e1, samplings, &kernel_matrix);
  data::Save("kde_kernel_matrix.csv", kernel_matrix);

  Matrix kernel_principal_components;
  KernelPCA(kernel_matrix, &kernel_principal_components);
  
  fx_done(fx_root);
}
  
