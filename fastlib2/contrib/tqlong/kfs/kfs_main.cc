
#include <fastlib/fastlib.h>
#include "kfs.h"

/** Create a 2-class problem where only the first two features matter */
void CreateSyntheticDataset(int D = 4, int N = 100) {
  Matrix data;
  data.Init(D+1,N);
  for (int k = 0; k <= D; k++) {
    if (k == D) { // labels
      for (int i = 0; i < N; i++) data.ref(k, i) = (i < N/2)? +1 : -1;
    }
    else if (k > 1) { // irrelevant features uniformly distributed in [-2, 2]
      for (int i = 0; i < N; i++) data.ref(k, i) = math::Random()*4-2;
    }
    else { // relevant features
      double mp[2] = {10, 7}, mn[2] = {7, 10};
      for (int i = 0; i < N; i++)
	data.ref(k, i) = math::Random()*4-2 + ((i < N/2)? mp[k]:mn[k]);
    }
  }
  data::Save("synthetic1.txt", data);
}

int main() {
  //CreateSyntheticDataset();
  return 0;
}
