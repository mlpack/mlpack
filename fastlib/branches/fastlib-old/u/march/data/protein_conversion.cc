#include "protein_conversion.h"

int main(int argc, char *argv[]) {
  
  fx_init(argc, argv);
  
  const char* data = fx_param_str(NULL, "data", "test.csv");
  const char* output_string = fx_param_str(NULL, "output", "default_output.csv");
  
  Dataset dataset;
  
  if (!PASSED(dataset.InitFromFile(data))) {
    fprintf(stderr, "You suck\n");
    return 1;
  }
  
  //printf("features: %d, points: %d\n", dataset.n_features(), dataset.n_points());
  
  index_t nFeats = dataset.n_features();
  index_t nPoints = dataset.n_points();
  
  Protein_Converter conv;
  conv.Init(nPoints, output_string);
  
  for (index_t i = 0; i < nPoints; i++) {
    
    double *temp = dataset.point(i);
    Matrix structure;
    structure.Copy(temp, 3, (nFeats/3));
    //structure.PrintDebug();
    conv.computeFeatures(structure);
    
  }
  
  conv.PrintData();
  
  fx_done();
  
  
}
