#include "fastlib/fastlib.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  const char *data = fx_param_str(NULL, "data", NULL);
  
  Matrix D;
  data::Load(data, &D);
  index_t n_features = D.n_rows();
  index_t n_points = D.n_cols();
  
  if (fx_param_bool(NULL, "remove_column", false)) {
    Matrix D_new;
    D_new.Init(n_features - 1, n_points);
    for (index_t j = 0; j < n_points; j++) {
      mem::Copy(D_new.GetColumnPtr(j), D.GetColumnPtr(j), n_features - 1);
    }
    D.Destruct();
    D.Own(&D_new);
  }
  
  Matrix U;
  Matrix VT;
  Vector s;
  
  fx_timer_start(NULL, "svd");
  la::SVDInit(D, &s, &U, &VT);
  fx_timer_stop(NULL, "svd");
  
  Matrix S_VT;
  fx_timer_start(NULL, "multiply");
  S_VT.Copy(VT);
  la::ScaleRows(s, &S_VT);
  fx_timer_stop(NULL, "multiply");
  
  Matrix S_row;
  S_row.AliasColVector(s);
  
  if (!fx_param_exists(NULL, "nosave")) {
    data::Save("S_VT.csv", S_VT);
    data::Save("S.csv", S_row);
    data::Save("U.csv", U);
  }
  
  fx_done();
  
  return 0;
}
