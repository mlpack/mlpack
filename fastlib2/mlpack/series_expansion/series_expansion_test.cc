#include "fastlib/fastlib.h"
#include "fastlib/base/test.h"
#include "fourier_expansion.h"
#include "fourier_series_expansion_aux.h"
#include "complex_matrix.h"
#include "fourier_kernel_aux.h"

class SeriesExpansionTest {

 public:

  void Init(fx_module *module_in) {
    module_ = module_in;
  }

  void TestFourierExpansion() {
    
    NOTIFY("[*] TestFourierExpansion");
    GaussianKernelFourierAux<double> kernel_aux;
    int order = 3;
    int dim = 3;
    double bandwidth = 0.1;
    kernel_aux.Init(bandwidth, order, dim);

    // Create an expansion from a random synthetic dataset.
    Matrix random_dataset;
    Vector weights;
    random_dataset.Init(20, 3);
    Vector center;
    center.Init(3);
    center.SetZero();
    weights.Init(20);
    for(index_t j = 0; j < 3; j++) {
      for(index_t i = 0; i < 20; i++) {
	random_dataset.set(i, j, math::Random(0, i));
      }
    }
    for(index_t i = 0; i < 20; i++) {
      la::AddTo(center.length(), random_dataset.GetColumnPtr(i), center.ptr());
    }
    la::Scale(center.length(), 1.0 / ((double) 20), center.ptr());
    
    weights.SetAll(1.0);
    FourierExpansion<GaussianKernelFourierAux<double> > expansion;
    expansion.Init(center, kernel_aux);
    
    expansion.AccumulateCoeffs(random_dataset, weights, 0, 20, 3);
    
    // Retrieve the coefficients and print them out.
    const ComplexVector<double> &coeffs = expansion.get_coeffs();
    coeffs.PrintDebug();
  }

  void TestFourierExpansionMapping() {

    NOTIFY("[*] TestFourierExpansionMapping");
    FourierSeriesExpansionAux<double> series_aux;
    int order = 2;
    int dim = 3;
    series_aux.Init(order, dim);
    series_aux.PrintDebug();

    // Verify that the shifting of each multiindex by the max order
    // roughly corresponds to the base ((2 * order) + 1) number.
    int total_num_mapping = (int) pow(2 * order + 1, dim);
    for(index_t i = 0; i < total_num_mapping; i++) {
      const ArrayList<short int> &mapping = series_aux.get_multiindex(i);

      int number = 0;
      printf("The mapping: ");
      for(index_t j = 0; j < dim; j++) {
	number = (2 * order + 1) * number + (mapping[j] + order);
	printf("%d ", mapping[j]);
      }
      printf("maps to %d\n", number);
      
      if(number != i) {
	FATAL("The mapping at the position %d is computed incorrectly!", i);
      }
    }
  }
  
  void TestAll() {
    TestFourierExpansionMapping();
    TestFourierExpansion();
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
