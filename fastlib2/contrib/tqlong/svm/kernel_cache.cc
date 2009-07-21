#include <fastlib/fastlib.h>
#include "svm.h"

namespace SVMLib {

#define CACHE_LIMIT 1000

Kernel::Kernel(const KernelFunction& kfunc, const Matrix& X) : X_(X) {
  n_points = X_.n_cols();
  kernel_stored = (n_points <= CACHE_LIMIT);
  kfunc_ = kfunc;
  kernel_diag.Init(n_points);
  if (kernel_stored) {
    full_kernel.Init(n_points, n_points);
    for (index_t i = 0; i < n_points; i++)
      for (index_t j = 0; j < n_points; j++)
	full_kernel.ref(i, j) = kernel_call(i, j);
    for (index_t i = 0; i < n_points; i++)
      kernel_diag[i] = full_kernel.get(i, i);
    //ot::Print(full_kernel);
    sub_kernel.Init(0, 0);
    col_index.Init();
    lru_col.Init();
  }
  else {
    n_cols = (CACHE_LIMIT*CACHE_LIMIT)/n_points;
    sub_kernel.Init(n_points, n_cols);
    col_index.Init(n_points);
    lru_col.Init(n_cols);
    for (index_t i = 0; i < n_points; i++) col_index[i] = -1;
    for (index_t i = 0; i < n_cols; i++) lru_col[i] = -1;
    for (index_t i = 0; i < n_points; i++) kernel_diag[i] = kernel_call(i, i);
    full_kernel.Init(0, 0);
    lru_ptr = 0;
  }
  n_loads = 0;
}
  
void Kernel::get_element(index_t i, index_t j, 
			 double& Kii, double& Kjj, double& Kij) {
  DEBUG_ASSERT(i >= 0 && i < X_.n_cols() && j >= 0 && j < X_.n_cols());
  Kii = kernel_diag[i];
  Kjj = kernel_diag[j];
  if (kernel_stored) {
    Kij = full_kernel.get(i, j);
  }
  else {
    if (col_index[i] >= 0)
      Kij = sub_kernel.get(j, col_index[i]);
    else if (col_index[j] >= 0)
      Kij = sub_kernel.get(i, col_index[j]);
    else 
      Kij = sub_kernel.get(i, loadColumn(j));
  }
}
  
void Kernel::get_column(index_t i, Vector* col_i) {
  if (kernel_stored) {
    full_kernel.MakeColumnVector(i, col_i);
  }
  else {
    if (col_index[i] >= 0)
      sub_kernel.MakeColumnVector(col_index[i], col_i);
    else
      sub_kernel.MakeColumnVector(loadColumn(i), col_i);
  }
}
  
void Kernel::get_diag(Vector* diag) {
  diag->Alias(kernel_diag);
}

double Kernel::kernel_call(index_t i, index_t j) {
  Vector col_i;
  Vector col_j;
  X_.MakeColumnVector(i, &col_i);
  X_.MakeColumnVector(j, &col_j);
  return kfunc_(col_i, col_j);
}

index_t Kernel::loadColumn(index_t i) {
  if (col_index[i] >= 0) return col_index[i];
  n_loads++;
  index_t delete_col = lru_col[lru_ptr], sub_index;
  if (delete_col == -1) { // not full --> find an empty column
    sub_index = lru_ptr;
  }
  else { // full --> have to delete
    sub_index = col_index[delete_col];
    col_index[delete_col] = -1;
  }
  col_index[i] = sub_index;

  for (index_t j = 0; j < n_points; j++)
    sub_kernel.ref(j, sub_index) = kernel_call(j, i);
  
  lru_col[lru_ptr] = i;
  lru_ptr = (lru_ptr+1) % n_cols;
  return sub_index;
}

}
