#include "multibody_force_problem.h"
#include "multibody_kernel.h"
#include "../multitree_template/multitree_dfs.h"
#include "../nested_summation_template/strata.h"

double AxilrodTellerForceKernelPositiveEvaluate
(int dimension_in, const double *first_point, const double *second_point,
 double first_distance, double first_distance_pow_three,
 double first_distance_pow_five, double first_distance_pow_seven,
 double second_distance, double second_distance_pow_three,
 double second_distance_pow_five, double third_distance,
 double third_distance_pow_three, double third_distance_pow_five) {
  
  double first_factor = second_distance / third_distance_pow_five *
    (first_point[dimension_in] - second_point[dimension_in]);
  double second_factor = third_distance / second_distance_pow_five *
    (first_point[dimension_in] - second_point[dimension_in]);
  return 1.875 * (first_factor + second_factor) /
    first_distance_pow_seven;
}

double AxilrodTellerForceKernelNegativeEvaluate
(int dimension_in, const double *first_point, const double *second_point,
 double first_distance, double first_distance_pow_three,
 double first_distance_pow_five, double first_distance_pow_seven,
 double second_distance, double second_distance_pow_three,
 double second_distance_pow_five, double third_distance,
 double third_distance_pow_three, double third_distance_pow_five) {
  
  double coord_diff = first_point[dimension_in] - 
    second_point[dimension_in];
  
  double first_factor = (-0.75) / 
    (first_distance_pow_five * second_distance_pow_three *
     third_distance_pow_three);
  double second_factor = (-0.375) /
    (first_distance * second_distance_pow_five * third_distance_pow_five);
  double third_factor = (-0.375) / 
    (first_distance_pow_three * second_distance_pow_three *
     third_distance_pow_five);
  double fourth_factor = (-0.375) /
    (first_distance_pow_three * second_distance_pow_five *
     third_distance_pow_three);
  double fifth_factor = (-1.125) /
    (first_distance_pow_five * second_distance * third_distance_pow_five);
  double sixth_factor = (-1.125) /
    (first_distance_pow_five * second_distance_pow_five * third_distance);
  double seventh_factor = (-1.875) /
    (first_distance_pow_seven * second_distance * third_distance_pow_three);
  double eigth_factor = (-1.875) /
    (first_distance_pow_seven * second_distance_pow_three * third_distance);
  
  // The eight negative components.
  return coord_diff * 
    (first_factor + second_factor + third_factor + fourth_factor +
     fifth_factor + sixth_factor + seventh_factor + eigth_factor);
}

void MultibodyBruteForce(const Matrix &particle_set) {

  Matrix short_bruteforce;
  short_bruteforce.Init(3, particle_set.n_cols());
  short_bruteforce.SetZero();
  Vector netforce;
  netforce.Init(3);
  netforce.SetZero();

  for(index_t i = 0; i < particle_set.n_cols() - 2; i++) {

    const double *first_point = particle_set.GetColumnPtr(i);
    double *first_point_force_vector = short_bruteforce.GetColumnPtr(i);

    for(index_t j = i + 1; j < particle_set.n_cols() - 1; j++) {

      const double *second_point = particle_set.GetColumnPtr(j);
      double *second_point_force_vector = short_bruteforce.GetColumnPtr(j);
      double squared_distance_between_i_and_j = 
	la::DistanceSqEuclidean(3, first_point, second_point) + DBL_EPSILON;
      double pow_squared_distance_between_i_and_j_0_5 =
	sqrt(squared_distance_between_i_and_j);
      double pow_squared_distance_between_i_and_j_1_5 =
	squared_distance_between_i_and_j *
	pow_squared_distance_between_i_and_j_0_5;
      double pow_squared_distance_between_i_and_j_2_5 =
	squared_distance_between_i_and_j *
	pow_squared_distance_between_i_and_j_1_5;
      double pow_squared_distance_between_i_and_j_3_5 =
	squared_distance_between_i_and_j *
	pow_squared_distance_between_i_and_j_2_5;
      

      for(index_t k = j + 1; k < particle_set.n_cols(); k++) {

	const double *third_point = particle_set.GetColumnPtr(k);
	double *third_point_force_vector = short_bruteforce.GetColumnPtr(k);
	double squared_distance_between_i_and_k =
	  la::DistanceSqEuclidean(3, first_point, third_point) + DBL_EPSILON;
	double squared_distance_between_j_and_k =
	  la::DistanceSqEuclidean(3, second_point, third_point) + DBL_EPSILON;

	double pow_squared_distance_between_i_and_k_0_5 =
	  sqrt(squared_distance_between_i_and_k);
	double pow_squared_distance_between_i_and_k_1_5 =
	  squared_distance_between_i_and_k *
	  pow_squared_distance_between_i_and_k_0_5;
	double pow_squared_distance_between_i_and_k_2_5 =
	  squared_distance_between_i_and_k *
	  pow_squared_distance_between_i_and_k_1_5;
	double pow_squared_distance_between_i_and_k_3_5 =
	  squared_distance_between_i_and_k *
	  pow_squared_distance_between_i_and_k_2_5;

	double pow_squared_distance_between_j_and_k_0_5 =
	  sqrt(squared_distance_between_j_and_k);
	double pow_squared_distance_between_j_and_k_1_5 =
	  squared_distance_between_j_and_k *
	  pow_squared_distance_between_j_and_k_0_5;
	double pow_squared_distance_between_j_and_k_2_5 =
	  squared_distance_between_j_and_k *
	  pow_squared_distance_between_j_and_k_1_5;
	double pow_squared_distance_between_j_and_k_3_5 =
	  squared_distance_between_j_and_k *
	  pow_squared_distance_between_j_and_k_2_5;

	for(index_t d = 0; d < 3; d++) {

	  double positive_contribution1 =
	    AxilrodTellerForceKernelPositiveEvaluate
	    (d, first_point, second_point,
	     pow_squared_distance_between_i_and_j_0_5,
	     pow_squared_distance_between_i_and_j_1_5,
	     pow_squared_distance_between_i_and_j_2_5,
	     pow_squared_distance_between_i_and_j_3_5,
	     pow_squared_distance_between_i_and_k_0_5,
	     pow_squared_distance_between_i_and_k_1_5,
	     pow_squared_distance_between_i_and_k_2_5,
	     pow_squared_distance_between_j_and_k_0_5,
	     pow_squared_distance_between_j_and_k_1_5,
	     pow_squared_distance_between_j_and_k_2_5);

	  double positive_contribution2 =
	    AxilrodTellerForceKernelPositiveEvaluate
	    (d, first_point, third_point,
	     pow_squared_distance_between_i_and_k_0_5,
	     pow_squared_distance_between_i_and_k_1_5,
	     pow_squared_distance_between_i_and_k_2_5,
	     pow_squared_distance_between_i_and_k_3_5,
	     pow_squared_distance_between_i_and_j_0_5,
	     pow_squared_distance_between_i_and_j_1_5,
	     pow_squared_distance_between_i_and_j_2_5,
	     pow_squared_distance_between_j_and_k_0_5,
	     pow_squared_distance_between_j_and_k_1_5,
	     pow_squared_distance_between_j_and_k_2_5);

	  double positive_contribution3 =
	    AxilrodTellerForceKernelPositiveEvaluate
	    (d, second_point, third_point,
	     pow_squared_distance_between_j_and_k_0_5,
	     pow_squared_distance_between_j_and_k_1_5,
	     pow_squared_distance_between_j_and_k_2_5,
	     pow_squared_distance_between_j_and_k_3_5,
	     pow_squared_distance_between_i_and_k_0_5,
	     pow_squared_distance_between_i_and_k_1_5,
	     pow_squared_distance_between_i_and_k_2_5,
	     pow_squared_distance_between_i_and_j_0_5,
	     pow_squared_distance_between_i_and_j_1_5,
	     pow_squared_distance_between_i_and_j_2_5);

	  double negative_contribution1 =
	    AxilrodTellerForceKernelNegativeEvaluate
	    (d, first_point, second_point,
	     pow_squared_distance_between_i_and_j_0_5,
	     pow_squared_distance_between_i_and_j_1_5,
	     pow_squared_distance_between_i_and_j_2_5,
	     pow_squared_distance_between_i_and_j_3_5,
	     pow_squared_distance_between_i_and_k_0_5,
	     pow_squared_distance_between_i_and_k_1_5,
	     pow_squared_distance_between_i_and_k_2_5,
	     pow_squared_distance_between_j_and_k_0_5,
	     pow_squared_distance_between_j_and_k_1_5,
	     pow_squared_distance_between_j_and_k_2_5);

	  double negative_contribution2 =
	    AxilrodTellerForceKernelNegativeEvaluate
	    (d, first_point, third_point,
	     pow_squared_distance_between_i_and_k_0_5,
	     pow_squared_distance_between_i_and_k_1_5,
	     pow_squared_distance_between_i_and_k_2_5,
	     pow_squared_distance_between_i_and_k_3_5,
	     pow_squared_distance_between_i_and_j_0_5,
	     pow_squared_distance_between_i_and_j_1_5,
	     pow_squared_distance_between_i_and_j_2_5,
	     pow_squared_distance_between_j_and_k_0_5,
	     pow_squared_distance_between_j_and_k_1_5,
	     pow_squared_distance_between_j_and_k_2_5);

	  double negative_contribution3 =
	    AxilrodTellerForceKernelNegativeEvaluate
	    (d, second_point, third_point,
	     pow_squared_distance_between_j_and_k_0_5,
	     pow_squared_distance_between_j_and_k_1_5,
	     pow_squared_distance_between_j_and_k_2_5,
	     pow_squared_distance_between_j_and_k_3_5,
	     pow_squared_distance_between_i_and_k_0_5,
	     pow_squared_distance_between_i_and_k_1_5,
	     pow_squared_distance_between_i_and_k_2_5,
	     pow_squared_distance_between_i_and_j_0_5,
	     pow_squared_distance_between_i_and_j_1_5,
	     pow_squared_distance_between_i_and_j_2_5);
	  
	  double first_point_contribution =
	    -(positive_contribution1 + negative_contribution1) -
	    (positive_contribution2 + negative_contribution2);
	  double second_point_contribution =
	    (positive_contribution1 + negative_contribution1) -
	    (positive_contribution3 + negative_contribution3);
	  double third_point_contribution =
	    (positive_contribution2 + negative_contribution2) +
	    (positive_contribution3 + negative_contribution3);
	  
	  first_point_force_vector[d] += first_point_contribution;
	  second_point_force_vector[d] += second_point_contribution;
	  third_point_force_vector[d] += third_point_contribution;
	}

      }
    }
  }

  FILE *foutput = fopen("short_bruteforce.txt", "w+");
  for(index_t i = 0; i < particle_set.n_cols(); i++) {

    for(index_t d = 0; d < 3; d++) {
      fprintf(foutput, "%g ", short_bruteforce.get(d, i));
    }
    la::AddTo(3, short_bruteforce.GetColumnPtr(i),
	      netforce.ptr());
    fprintf(foutput, "\n");
  }
  netforce.PrintDebug("", stdout);
  fclose(foutput);
}

int main(int argc, char *argv[]) {

  // Initialize FastExec (parameter handling stuff).
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "fmm_module" under the
  // root directory (NULL) for the multibody object to work inside.
  // Here, we initialize it with all parameters defined
  // "--multibody/...=...".
  struct datanode *multibody_module = fx_submodule(fx_root, "multibody");
  
  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");

  // Query and reference datasets, reference weight dataset.
  Matrix references;
  
  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);


  la::Scale(references.n_cols() * references.n_rows(), 30.0, references.ptr());

  // Instantiate a multi-tree problem...
  ArrayList<const Matrix *> sets;
  sets.Init(AxilrodTellerForceProblem::order);
  for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
    sets[i] = &references;
  }

  MultiTreeDepthFirst<AxilrodTellerForceProblem> algorithm;
  AxilrodTellerForceProblem::MultiTreeQueryResult results;
  AxilrodTellerForceProblem::MultiTreeQueryResult naive_results;
  algorithm.InitMonoChromatic(sets, (const ArrayList<const Matrix *> *) NULL,
			      multibody_module);

  fx_timer_start(fx_root, "multitree");
  algorithm.Compute(NULL, &results, true);
  fx_timer_stop(fx_root, "multitree");

  results.PrintDebug("force_vectors.txt");
  printf("Got %d finite difference prunes...\n",
	 results.num_finite_difference_prunes);
  printf("Got %d Monte Carlo prunes...\n", results.num_monte_carlo_prunes);
  
  fx_timer_start(fx_root, "naive_code");
  algorithm.NaiveCompute((const ArrayList<const Matrix *> *) NULL,
			 &naive_results);
  fx_timer_stop(fx_root, "naive_code");
  naive_results.PrintDebug("naive_force_vectors.txt");
  double max_relative_error, positive_max_relative_error,
    negative_max_relative_error;
  naive_results.MaximumRelativeError(results, &max_relative_error,
				     &negative_max_relative_error,
				     &positive_max_relative_error);
  printf("Maximum relative error: %g\n", max_relative_error);
  printf("Positive max relative error: %g\n", positive_max_relative_error);
  printf("Negative max relative error: %g\n", negative_max_relative_error);


  // MultibodyBruteForce(references);

  fx_done(fx_root);
  return 0;
}
