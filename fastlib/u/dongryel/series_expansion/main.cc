/**
 * @author Dongryeol Lee
 * @file main.cc
 *
 * Test driver for the Cartesian series expansion.
 */

#include "fastlib/fastlib.h"
#include "series_expansion.h"
#include "series_expansion_aux.h"

int TestEvaluateFarField(const Matrix &data, const Vector &weights,
			 const ArrayList<int> &rows) {
  
  printf("\n----- TestEvaluateFarField -----\n");

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
  SeriesExpansion se;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.5
  se.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::FARFIELD, center,
          sea.get_max_total_num_coeffs(), 0.5);

  // compute up to 4-th order multivariate polynomial.
  se.ComputeFarFieldCoeffs(data, weights, rows, 10, sea);

  // print out the objects
  se.PrintDebug();               // expansion at (0, 0)

  // evaluate the series expansion
  printf("Evaluated the expansion at (%g %g) is %g...\n",
	 evaluate_here[0], evaluate_here[1],
	 se.EvaluateFarField(NULL, -1, &evaluate_here, &sea));

  // check with exhaustive method
  double exhaustive_sum = 0;
  for(index_t i = 0; i < rows.size(); i++) {
    int row_num = rows[i];
    
    exhaustive_sum += exp(-((evaluate_here[0] - data.get(0, row_num)) * 
			    (evaluate_here[0] - data.get(0, row_num)) +
			    (evaluate_here[1] - data.get(1, row_num)) * 
			    (evaluate_here[1] - data.get(1, row_num))) 
			  / (2 * 0.5));
  }
  printf("Exhaustively evaluated sum: %g\n", exhaustive_sum);
  return 1;
}

int TestEvaluateLocalField(const Matrix &data, const Vector &weights,
			   const ArrayList<int> &rows) {

  printf("\n----- TestEvaluateLocalField -----\n");

  // declare auxiliary object and initialize
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
  SeriesExpansion se;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 1
  se.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::LOCAL, center,
          sea.get_max_total_num_coeffs(), 1);

  // compute up to 4-th order multivariate polynomial.
  se.ComputeLocalCoeffs(data, weights, rows, 6, sea);

  // print out the objects
  se.PrintDebug();

  // evaluate the series expansion
  printf("Evaluated the expansion at (%g %g) is %g...\n",
	 evaluate_here[0], evaluate_here[1],
         se.EvaluateLocalField(NULL, -1, &evaluate_here, &sea));
  
  // check with exhaustive method
  double exhaustive_sum = 0;
  for(index_t i = 0; i < rows.size(); i++) {
    int row_num = rows[i];

    exhaustive_sum += exp(-((evaluate_here[0] - data.get(0, row_num)) *
                            (evaluate_here[0] - data.get(0, row_num)) +
                            (evaluate_here[1] - data.get(1, row_num)) *
                            (evaluate_here[1] - data.get(1, row_num)))
                          / (2 * 1));
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
		      const ArrayList<int> &rows) {

  printf("\n----- TestTransFarToFar -----\n");

  // declare auxiliary object and initialize
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
  SeriesExpansion se;
  SeriesExpansion se_translated;
  SeriesExpansion se_cmp;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.1
  se.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::FARFIELD, center, 
	  sea.get_max_total_num_coeffs(), 0.1);
  se_translated.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::FARFIELD, 
		     new_center, sea.get_max_total_num_coeffs(), 0.1);
  se_cmp.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::FARFIELD, 
	      new_center, sea.get_max_total_num_coeffs(), 0.1);
  
  // compute up to 4-th order multivariate polynomial and translate it.
  se.ComputeFarFieldCoeffs(data, weights, rows, 4, sea);
  se_translated.TransFarToFar(se, sea);
  
  // now compute the same thing at (2, -2) and compare
  se_cmp.ComputeFarFieldCoeffs(data, weights, rows, 4, sea);

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
			  const ArrayList<int> &rows) {
  
  printf("\n----- TestTransLocalToLocal -----\n");

  // declare auxiliary object and initialize
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
  SeriesExpansion se;
  SeriesExpansion se_translated;
  SeriesExpansion se_cmp;

  // initialize expansion objects with respective centers and the bandwidth
  // squared of 0.1
  se.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::LOCAL, center, 
	  sea.get_max_total_num_coeffs(), 1);
  se_translated.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::LOCAL,
		     new_center, sea.get_max_total_num_coeffs(), 1);
  se_cmp.Init(SeriesExpansion::GAUSSIAN, SeriesExpansion::LOCAL,
	      new_center, sea.get_max_total_num_coeffs(), 1);
  
  // compute up to 4-th order multivariate polynomial and translate it.
  se.ComputeLocalCoeffs(data, weights, rows, 5, sea);
  se_translated.TransLocalToLocal(se, sea);
  
  // now compute the same thing at (1, -1) and compare
  se_cmp.ComputeLocalCoeffs(data, weights, rows, 5, sea);

  // print out the objects
  se.PrintDebug();               // expansion at (0, 0)
  se_translated.PrintDebug();    // expansion at (1, -1) translated from
                                 // one above
  se_cmp.PrintDebug();           // directly computed expansion at (1, -1)

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
      printf("Differs by %g at position %d...\n", 
	     fabs(se_translated_coeffs[i] - se_cmp_coeffs[i]) / 
	     fabs(se_cmp_coeffs[i]), i);
      return 0;
    }
  }
  
  return 1;
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *datafile_name = fx_param_str(NULL, "data", NULL);
  Dataset dataset;
  Matrix data;
  Vector weights;
  ArrayList<int> rows;
  
  // read the dataset and get the matrix
  if (!PASSED(dataset.InitFromFile(datafile_name))) {
    fprintf(stderr, "main: Couldn't open file '%s'.\n", datafile_name);
    return 1;
  }
  data.Alias(dataset.matrix());
  weights.Init(data.n_cols());
  weights.SetAll(1);
  rows.Init(data.n_cols());

  for(index_t i = 0; i < data.n_cols(); i++) {
    rows[i] = i;
  }


  // unit tests begin here!
  DEBUG_ASSERT(TestInitAux(data) == 1);
  DEBUG_ASSERT(TestEvaluateFarField(data, weights, rows) == 1);
  DEBUG_ASSERT(TestEvaluateLocalField(data, weights, rows) == 1);
  DEBUG_ASSERT(TestTransFarToFar(data, weights, rows) == 1);
  DEBUG_ASSERT(TestTransLocalToLocal(data, weights, rows) == 1);

  fx_done();
}
