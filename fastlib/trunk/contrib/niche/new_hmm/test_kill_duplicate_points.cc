#include "fastlib/fastlib.h"
#include "utils.h"

int main(int argc, char* argv[]) {

  Matrix data;

  int n_dims = 3;
  int n_points = 10;
  data.Init(n_dims, n_points);
  
  Vector x, y;
  x.Init(n_dims);
  y.Init(n_dims);

  for(int i = 0; i < n_dims; i++) {
    x[i] = drand48();
    y[i] = drand48();
  }

  for(int i = 0; i < 6; i++) {
    data.CopyVectorToColumn(i, x);
  }
  for(int i = 6; i < n_points; i++) {
    data.CopyVectorToColumn(i, y);
  }

  data.PrintDebug("data");

  Matrix new_data;
  Matrix weights;
  KillDuplicatePoints(data, &new_data, &weights);

  data.PrintDebug("data");
  new_data.PrintDebug("new data");
  weights.PrintDebug("weights");



}
