#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"
#include "fourier_series_expansion_aux.h"

class SeriesExpansionTest {

 public:

  void Init(fx_module *module_in) {
    module_ = module_in;
  }

  void TestFourierExpansionMapping() {

    FourierSeriesExpansionAux series_aux;
    series_aux.Init(2, 3);
    series_aux.PrintDebug();
  }

  void TestAll() {
    TestFourierExpansionMapping();
    NOTIFY("[*] All tests passed !!");
  }

 private:

  fx_module *module_;

};

int main(int argc, char *argv[]) {
  fx_module *module = fx_init(argc, argv, NULL);
  SeriesExpansionTest test;
  test.Init(module);
  test.TestAll();
  fx_done(module);
  return 0;
}
