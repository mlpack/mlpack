#include "emst.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char* data = fx_param_str(NULL, "data", "test.txt");
  
  int leafsize;
  leafsize = fx_param_int(NULL, "leafsize", 1);
  
  Dataset dataset;
  
  if (!PASSED(dataset.InitFromFile(data))) {
    fprintf(stderr, "Couldn't open file '%s'.\n", data);
    return 1;
  }
  
  Matrix points;
  points.Copy(dataset.matrix());
    
  
  EmstTree tree;
  tree.Init(points, leafsize);
  
    
  fx_done();

}
