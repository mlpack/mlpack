#include "fastlib/fastlib.h"
#include "lin_alg.h"

int main(int argc, char *argv[]) {
  Matrix base_matrix;
  base_matrix.Init(5,2);

  index_t num = 1;
  for(index_t i = 0; i < 5; i++) {
    for(index_t j = 0; j < 2; j++) {
      base_matrix.set(i, j, num);
      num++;
    }
  }

  Matrix new_matrix;

  RepeatMatrix(2, 5, base_matrix, &new_matrix);

  base_matrix.PrintDebug("bob");
  new_matrix.PrintDebug("sum");
  
  return SUCCESS_PASS;
}
