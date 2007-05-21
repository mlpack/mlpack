#include "nbr_utils.h"

success_t nbr_utils::Load(const char *fname,
    TempCacheArray<Vector> *cache_out, index_t vectors_per_block) {
  Matrix matrix;
  Vector first_row;
  success_t success = data::Load(fname, &matrix);
  
  matrix.MakeColumnVector(0, &first_row);
  cache_out->Init(first_row, matrix.n_cols(),
      max(1, min(matrix.n_cols(), vectors_per_block)));
  
  for (index_t i = 0; i < matrix.n_cols(); i++) {
    Vector *dest_vector = cache_out->StartWrite(i);
    dest_vector->CopyValues(matrix.GetColumnPtr(i));
    cache_out->StopWrite(i);
  }
  
  return success;
}
