/**
 * @author Dongryeol Lee
 * @file main.cc
 *
 * Test driver for the Cartesian series expansion.
 */

#include "fastlib/fastlib.h"
#include "kernel_derivative.h"
#include "farfield_expansion.h"
#include "local_expansion.h"
#include "series_expansion_aux.h"

int TestEpanKernelEvaluateFarField(const Matrix &data, const Vector &weights,
				   int begin, int end) {
  
  printf("\n----- TestEpanKernelEvaluateFarField -----\n");

  // bandwidth of 10 Epanechnikov kernel
  double bandwidth = 10;
  EpanKernel kernel;
  kernel.Init(bandwidth);

  // declare auxiliary object and initialize
  SeriesExpansionAux sea;
  sea.Init(10, data.n_rows());

  // declare center at the origin
  Vector center;
  center.Init(2);
  center.SetZero();

  // to-be-evaluated point
  Vector evaluate_here;
  evaluate_here.Init(2);
  evaluate_here[0] = evaluate_here[1] = 0.1;

  // declare expansion object
  FarFieldExpansion<EpanKernel, EpanKernelDerivative> se;

  // initialize expansion objects with respective center and the bandwidth
  se.Init(bandwidth, center, &sea);

  // compute up to 2-nd order multivariate polynomial.
  se.AccumulateCoeffs(data, weights, begin, end, 2);

  // print out the objects
  se.PrintDebug();               // expansion at (0, 0)

  // evaluate the series expansion
  printf("Evaluated the expansion at (%g %g) is %g...\n",
	 evaluate_here[0], evaluate_here[1],
	 se.EvaluateField(NULL, -1, &evaluate_here));

  // check with exhaustive method
  double exhaustive_sum = 0;
  for(index_t i = begin; i < end; i++) {
    int row_num = i;
    double dsqd = (evaluate_here[0] - data.get(0, row_num)) * 
      (evaluate_here[0] - data.get(0, row_num)) +
      (evaluate_here[1] - data.get(1, row_num)) * 
      (evaluate_here[1] - data.get(1, row_num));
    
    exhaustive_sum += kernel.EvalUnnormOnSq(dsqd);

  }
  printf("Exhaustively evaluated sum: %g\n", exhaustive_sum);

  // now recompute using the old expansion formula...
  double first_moment0 = 0;
  double first_moment1 = 0;
  double second_moment0 = 0;
  double second_moment1 = 0;

  for(index_t i = begin; i < end; i++) {
    int row_num = i;

    double diff0 = (data.get(0, row_num) - center[0]) / 
      sqrt(kernel.bandwidth_sq());
    double diff1 = (data.get(1, row_num) - center[1]) / 
      sqrt(kernel.bandwidth_sq());

    first_moment0 += diff0;
    first_moment1 += diff1;
    second_moment0 += diff0 * diff0;
    second_moment1 += diff1 * diff1;
  }
  double diff_coord0 = (evaluate_here[0] - center[0]) / 
    sqrt(kernel.bandwidth_sq());
  double diff_coord1 = (evaluate_here[1] - center[1]) / 
    sqrt(kernel.bandwidth_sq());
  
  printf("Old formula got: %g\n",
	 end - (second_moment0 - 2 * first_moment0 * diff_coord0 + 
		end * diff_coord0 * diff_coord0) - 
	 (second_moment1 - 2 * first_moment1 * diff_coord1 + 
	  end * diff_coord1 * diff_coord1));

  return 1;
}

int TestEvaluateFarField(const Matrix &data, const Vector &weights,
			 int begin, int end) {
  
  printf("\n----- TestEvaluateFarField -----\n");

  // bandwidth of sqrt(0.5) Gaussian kernel
  double bandwidth = sqrt(0.5);
  GaussianKernel kernel;
  kernel.Init(sqrt(0.5));

  // declare auxiliary object and initialize
  SeriesExpansionAux sea;
  sea.Init(10, data.n_rows());

  // declare center at the origin
  Vector center;
  center.Init(2);
  center.SetZero();

  // to-be-evaluated point
  Vector evaluate_here;
  evaluate_here.Init(2);
  evaluate_here[0] = evaluate_here[1] = 3;

  // declare expansion objects at (0,0) and other centers
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.5
  se.Init(bandwidth, center, &sea);

  // compute up to 4-th order multivariate polynomial.
  se.AccumulateCoeffs(data, weights, begin, end, 10);

  // print out the objects
  se.PrintDebug();               // expansion at (0, 0)

  // evaluate the series expansion
  printf("Evaluated the expansion at (%g %g) is %g...\n",
	 evaluate_here[0], evaluate_here[1],
	 se.EvaluateField(NULL, -1, &evaluate_here));

  // check with exhaustive method
  double exhaustive_sum = 0;
  for(index_t i = begin; i < end; i++) {
    int row_num = i;
    double dsqd = (evaluate_here[0] - data.get(0, row_num)) * 
      (evaluate_here[0] - data.get(0, row_num)) +
      (evaluate_here[1] - data.get(1, row_num)) * 
      (evaluate_here[1] - data.get(1, row_num));
    
    exhaustive_sum += kernel.EvalUnnormOnSq(dsqd);

  }
  printf("Exhaustively evaluated sum: %g\n", exhaustive_sum);
  return 1;
}

int TestEvaluateLocalField(const Matrix &data, const Vector &weights,
			   int begin, int end) {

  printf("\n----- TestEvaluateLocalField -----\n");

  // declare auxiliary object and initialize
  double bandwidth = 1;
  GaussianKernel kernel;
  kernel.Init(bandwidth);
  SeriesExpansionAux sea;
  sea.Init(10, data.n_rows());

  // declare center at the origin
  Vector center;
  center.Init(2);
  center[0] = center[1] = 4;

  // to-be-evaluated point
  Vector evaluate_here;
  evaluate_here.Init(2);
  evaluate_here[0] = evaluate_here[1] = 3.5;

  // declare expansion objects at (0,0) and other centers
  LocalExpansion<GaussianKernel, GaussianKernelDerivative> se;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 1
  se.Init(bandwidth, center, &sea);

  // compute up to 4-th order multivariate polynomial.
  se.AccumulateCoeffs(data, weights, begin, end, 6);

  // print out the objects
  se.PrintDebug();

  // evaluate the series expansion
  printf("Evaluated the expansion at (%g %g) is %g...\n",
	 evaluate_here[0], evaluate_here[1],
         se.EvaluateField(NULL, -1, &evaluate_here));
  
  // check with exhaustive method
  double exhaustive_sum = 0;
  for(index_t i = begin; i < end; i++) {
    int row_num = i;

    double dsqd = (evaluate_here[0] - data.get(0, row_num)) * 
      (evaluate_here[0] - data.get(0, row_num)) +
      (evaluate_here[1] - data.get(1, row_num)) * 
      (evaluate_here[1] - data.get(1, row_num));
    
    exhaustive_sum += kernel.EvalUnnormOnSq(dsqd);

  }
  printf("Exhaustively evaluated sum: %g\n", exhaustive_sum);
  return 1;
}

int TestInitAux(const Matrix& data) {

  printf("\n----- TestInitAux -----\n");

  SeriesExpansionAux sea;
  sea.Init(10, data.n_rows());
  sea.PrintDebug();

  return 1;
}

int TestTransFarToFar(const Matrix &data, const Vector &weights,
		      int begin, int end) {

  printf("\n----- TestTransFarToFar -----\n");

  // declare auxiliary object and initialize
  double bandwidth = sqrt(0.1);
  SeriesExpansionAux sea;
  sea.Init(10, data.n_rows());
  
  // declare center at the origin
  Vector center;
  center.Init(2);
  center.SetZero();

  // declare a new center at (2, -2)
  Vector new_center;
  new_center.Init(2);
  new_center[0] = 2;
  new_center[1] = -2;

  // declare expansion objects at (0,0) and other centers
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se;
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se_translated;
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se_cmp;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.1
  se.Init(bandwidth, center, &sea);
  se_translated.Init(bandwidth, new_center, &sea);
  se_cmp.Init(bandwidth, new_center, &sea);
  
  // compute up to 4-th order multivariate polynomial and translate it.
  se.AccumulateCoeffs(data, weights, begin, end, 4);
  se_translated.TranslateFromFarField(se);
  
  // now compute the same thing at (2, -2) and compare
  se_cmp.AccumulateCoeffs(data, weights, begin, end, 4);

  // print out the objects
  se.PrintDebug();               // expansion at (0, 0)
  se_translated.PrintDebug();    // expansion at (2, -2) translated from
                                 // one above
  se_cmp.PrintDebug();           // directly computed expansion at (2, -2)

  // retrieve the coefficients of se_translated and se_cmp
  Vector se_translated_coeffs;
  Vector se_cmp_coeffs;
  se_translated_coeffs.Alias(se_translated.get_coeffs());
  se_cmp_coeffs.Alias(se_cmp.get_coeffs());

  // assert that se_translated and se_cmp have equal set of coefficients
  // within 0.1 % error taking account the numerical errors
  for(index_t i = 0; i < se_cmp_coeffs.length(); i++) {
    if(fabs(se_translated_coeffs[i] - se_cmp_coeffs[i]) > 
       0.001 * fabs(se_cmp_coeffs[i])) {
      return 0;
    }
  }
  
  return 1;
}

int TestTransLocalToLocal(const Matrix &data, const Vector &weights,
			  int begin, int end) {
  
  printf("\n----- TestTransLocalToLocal -----\n");

  // declare auxiliary object and initialize
  double bandwidth = sqrt(1);
  SeriesExpansionAux sea;
  sea.Init(10, data.n_rows());
  
  // declare center at the origin
  Vector center;
  center.Init(2);
  center[0] = center[1] = 4;

  // declare a new center at (1, -1)
  Vector new_center;
  new_center.Init(2);
  new_center[0] = new_center[1] = 3.5;

  // declare expansion objects at (0,0) and other centers
  LocalExpansion<GaussianKernel, GaussianKernelDerivative> se;
  LocalExpansion<GaussianKernel, GaussianKernelDerivative> se_translated;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.1
  se.Init(bandwidth, center, &sea);
  se_translated.Init(bandwidth, new_center, &sea);
  
  // compute up to 4-th order multivariate polynomial and translate it.
  se.AccumulateCoeffs(data, weights, begin, end, 4);
  se.TranslateToLocal(se_translated);

  // print out the objects
  se.PrintDebug();               // expansion at (4, 4)
  se_translated.PrintDebug();    // expansion at (3.5, 3.5) translated from
                                 // one above

  // evaluate the expansion
  Vector evaluate_here;
  evaluate_here.Init(2);
  evaluate_here[0] = evaluate_here[1] = 3.75;
  double original_sum = se.EvaluateField(NULL, -1, &evaluate_here);
  double translated_sum = 
    se_translated.EvaluateField(NULL, -1, &evaluate_here);

  printf("Evaluating both expansions at (%g %g)...\n", evaluate_here[0],
	 evaluate_here[1]);
  printf("Sum evaluated at the original local expansion: %g\n", original_sum);
  printf("Sum evaluated at the translated local expansion: %g\n",
	 translated_sum);

  if(fabs(original_sum - translated_sum) > 0.001 * fabs(original_sum)) {
    return 0;
  }
  return 1;
}

int TestConvolveFarField(const Matrix &data, const Vector &weights,
			 int begin, int end) {
  
  printf("\n----- TestConvolveFarField -----\n");

  // bandwidth of 5 Gaussian kernel
  double bandwidth = 5;
  GaussianKernel kernel;
  kernel.Init(bandwidth);

  // declare auxiliary object and initialize
  SeriesExpansionAux sea;
  sea.Init(20, data.n_rows());

  // declare center at the origin, (10, -10) and (-10, -10)
  Vector center;
  center.Init(2);
  center.SetZero();
  Vector center2;
  center2.Init(2);
  center2[0] = 10; center2[1] = -10;
  Vector center3;
  center3.Init(2);
  center3[0] = center3[1] = -10;

  // create fake data
  Matrix data2, data3;
  data2.Copy(data);
  data3.Copy(data);
  for(index_t c = 0; c < data.n_cols(); c++) {
    data2.set(0, c, data2.get(0, c) + center2[0]); 
    data2.set(1, c, data2.get(1, c) + center2[1]);
    data3.set(0, c, data3.get(0, c) + center3[0]);
    data3.set(1, c, data3.get(1, c) + center3[1]);
  }

  data.PrintDebug();
  data2.PrintDebug();
  data3.PrintDebug();

  // declare expansion objects at (0,0) and other centers
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se;
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se2;
  FarFieldExpansion<GaussianKernel, GaussianKernelDerivative> se3;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.5
  se.Init(bandwidth, center, &sea);
  se2.Init(bandwidth, center2, &sea);
  se3.Init(bandwidth, center3, &sea);

  // compute up to 20-th order multivariate polynomial.
  se.AccumulateCoeffs(data, weights, begin, end, 20);
  se2.AccumulateCoeffs(data2, weights, begin, end, 20);
  se3.AccumulateCoeffs(data3, weights, begin, end, 20);

  printf("Convolution: %g\n", se.ConvolveField(se2, se3, 6, 6, 6));

  // compare with naive
  double naive_result = 0;
  for(index_t i = 0; i < data.n_cols(); i++) {
    const double *i_col = data.GetColumnPtr(i);
    for(index_t j = 0; j < data2.n_cols(); j++) {
      const double *j_col = data2.GetColumnPtr(j);
      for(index_t k = 0; k < data3.n_cols(); k++) {
	const double *k_col = data3.GetColumnPtr(k);

	// compute pairwise distances
	double dsqd_ij = 0;
	double dsqd_ik = 0;
	double dsqd_jk = 0;
	for(index_t d = 0; d < sea.get_dimension(); d++) {
	  dsqd_ij += (i_col[d] - j_col[d]) * (i_col[d] - j_col[d]);
	  dsqd_ik += (i_col[d] - k_col[d]) * (i_col[d] - k_col[d]);
	  dsqd_jk += (j_col[d] - k_col[d]) * (j_col[d] - k_col[d]);
	}
	naive_result += kernel.EvalUnnormOnSq(dsqd_ij + dsqd_ik +
					      dsqd_jk);
      }
    }
  }
  printf("Naive algorithm: %g\n", naive_result);
  return 1;
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *datafile_name = fx_param_str(NULL, "data", NULL);
  Dataset dataset;
  Matrix data;
  Vector weights;
  int begin, end;
  
  // read the dataset and get the matrix
  if (!PASSED(dataset.InitFromFile(datafile_name))) {
    fprintf(stderr, "main: Couldn't open file '%s'.\n", datafile_name);
    return 1;
  }
  data.Alias(dataset.matrix());
  weights.Init(data.n_cols());
  weights.SetAll(1);
  begin = 0; end = data.n_cols();

  // unit tests begin here!
  DEBUG_ASSERT(TestInitAux(data) == 1);
  DEBUG_ASSERT(TestEvaluateFarField(data, weights, begin, end) == 1);
  DEBUG_ASSERT(TestEvaluateLocalField(data, weights, begin, end) == 1);
  DEBUG_ASSERT(TestTransFarToFar(data, weights, begin, end) == 1);
  DEBUG_ASSERT(TestTransLocalToLocal(data, weights, begin, end) == 1);

  DEBUG_ASSERT(TestEpanKernelEvaluateFarField(data, weights, begin, end) == 1);

  DEBUG_ASSERT(TestConvolveFarField(data, weights, begin, end) == 1);

  fx_done();
}
