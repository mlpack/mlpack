#include <fastlib/fastlib.h>
#include "svm.h"

namespace SVMLib {

IndexSet::IndexSet(index_t n) {
  n_total = n;
  n_set = 0;
  index_set.Init(n);
  set_index.Init(n);
  InitEmpty();
}

void IndexSet::InitEmpty() {
  for (index_t i = 0; i < n_total; i++) set_index[i] = -1;
  //index_set.Clear();
}
 
void IndexSet::addremove(index_t i, bool b) {
  DEBUG_ASSERT(i < n_total);
  // adding
  if (b && set_index[i] == -1) {
    index_set[n_set] = i;
    set_index[i] = n_set;
    n_set++;
  }
  // removing
  if (!b && set_index[i] != -1) {
    index_t index = set_index[i];
    set_index[i] = -1;
    if (index < n_set-1) {
      index_set[index] = index_set[n_set-1]; // move the last point to index position
      set_index[index_set[n_set-1]] = index; // set the new index
    }
    n_set--;
  }
}

void IndexSet::print() { 
  for (index_t i = 0; i < n_set; i++)
    printf("%16d", index_set[i]);
  printf("\n");
}

void IndexSet::print(const Vector& x) {
  DEBUG_ASSERT(x.length() == n_total);
  for (index_t i = 0; i < n_set; i++)
    printf("%5d:%11f", index_set[i],x[index_set[i]]);
  printf("\n");
}

};
