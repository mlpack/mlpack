/** @file distributed_kpca_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_DEV_H
#define MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_DEV_H

#include <boost/scoped_array.hpp>
#include "core/metric_kernels/lmetric.h"
#include "core/monte_carlo/mean_variance_pair_matrix.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/transform.h"
#include "mlpack/distributed_kpca/distributed_kpca.h"
#include "mlpack/distributed_kpca/distributed_kpca_argument_parser_dev.h"
#include "mlpack/distributed_kpca/threaded_convergence.h"
#include "mlpack/series_expansion/random_feature.h"

namespace core {
namespace parallel {

/** @brief The reduction function for adding two dense points
 *         together coordinate-wise.
 */
class AddDensePoint:
  public std::binary_function <
  arma::vec,
  arma::vec,
    arma::vec > {

  public:
    const arma::vec operator()(
      const arma::vec &a,
      const arma::vec &b) const {

      arma::vec sum;
      sum = a + b;
      return sum;
    }
};

/** @brief The reduction function for combining two Monte Carlo
 *         estimates.
 */
class CombineMeanVariancePairMatrix:
  public std::binary_function <
  core::monte_carlo::MeanVariancePairMatrix,
  core::monte_carlo::MeanVariancePairMatrix,
    core::monte_carlo::MeanVariancePairMatrix > {

  public:
    const core::monte_carlo::MeanVariancePairMatrix operator()(
      const core::monte_carlo::MeanVariancePairMatrix &a,
      const core::monte_carlo::MeanVariancePairMatrix &b) const {

      core::monte_carlo::MeanVariancePairMatrix combined;
      combined.Init(a.n_rows(), a.n_cols());
      combined.set_total_num_terms(a.get(0, 0).total_num_terms());
      combined.CopyValues(a);
      combined.CombineWith(b);

      return combined;
    }
};
}
}

namespace boost {
namespace mpi {

/** @brief Indicates that the function for adding two dense points
 *         is a commutative reduction operator.
 */
template<>
class is_commutative <
  core::parallel::AddDensePoint,
  arma::vec  > :
  public boost::mpl::true_ {

};

/** @brief Indicates that the function for combining Monte Carlo
 *         sampling results is a commutative reduction operator.
 */
template<>
class is_commutative <
  core::parallel::CombineMeanVariancePairMatrix,
  core::monte_carlo::MeanVariancePairMatrix  > :
  public boost::mpl::true_ {

};
}
}

namespace core {
namespace table {

// The extern declaration for the memory mapped file.
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_kpca {

template<typename DistributedTableType, typename KernelType>
DistributedKpca<DistributedTableType, KernelType>::DistributedKpca() {
  world_ = NULL;
  mult_const_ = 0.0;
  effective_num_reference_points_ = 0;
  correction_term_ = 0.0;
  max_num_iterations_ = 0;
  num_random_fourier_features_eigen_ = 1;
  num_dimensions_ = 0;
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::GenerateRandomFourierFeatures_(
  int num_random_fourier_features,
  arma::mat *random_variates,
  boost::scoped_array<arma::vec> &random_variate_aliases) {

  random_variates->set_size(num_dimensions_, num_random_fourier_features);
  if(world_->rank() == 0) {
    for(int i = 0; i < num_random_fourier_features; i++) {

      // Draw a random Fourier feature.
      arma::vec point_alias;
      core::table::MakeColumnVector(*random_variates, i, &point_alias);
      kernel_.DrawRandomVariate(
        num_dimensions_, & point_alias);
    }
  }

  // The master broadcasts the set of random Fourier features. All
  // processes form a set of armadillo vector aliases.
  boost::mpi::broadcast(*world_, *random_variates, 0);
  for(int i = 0; i < num_random_fourier_features; i++) {
    core::table::MakeColumnVector(
      *random_variates, i, &(random_variate_aliases[i]));
  }
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::FinalizeKernelEigenvectors_(
  double num_standard_deviations,
  int num_reference_samples,
  const mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > &arguments_in,
  mlpack::distributed_kpca::KpcaResult *result_out) {

  // Temporary space for performing the covariance eigenvector times
  // the projected reference set.
  arma::mat &covariance_eigenvectors_alias =
    result_out->covariance_eigenvectors();
  arma::mat &reference_projections_alias =
    result_out->reference_projections();
  arma::mat product = arma::trans(covariance_eigenvectors_alias) *
                      reference_projections_alias;

  // Normalize each KPCA components. Each process needs to compute its
  // squared length and participate in the global all-reduce step.
  arma::vec local_squared_length_contributions;
  local_squared_length_contributions.zeros(product.n_rows);
  arma::vec global_lengths;
  for(unsigned int j = 0; j < product.n_cols; j++) {
    for(unsigned int i = 0; i < product.n_rows; i++) {
      local_squared_length_contributions[i] +=
        core::math::Sqr(product.at(i, j));
    }
  }
  boost::mpi::all_reduce(
    *world_, local_squared_length_contributions,
    global_lengths, core::parallel::AddDensePoint());

  // Take the square roots of each KPCA component to compute the
  // actual normalization factor.
  std::vector< std::pair<int, double> > eigenvalues(global_lengths.n_elem);
  for(unsigned int i = 0; i < global_lengths.n_elem; i++) {
    eigenvalues[i].first = i;
    eigenvalues[i].second =
      global_lengths[i] / static_cast<double>(product.n_cols);
    global_lengths[i] = sqrt(global_lengths[i]);
  }

  // Sort the eigenvalues.
  std::sort(
    eigenvalues.begin(), eigenvalues.end(),
    boost::bind(&std::pair<int, double>::second, _1) >
    boost::bind(&std::pair<int, double>::second, _2));

  // Copy the eigenvectors out.
  result_out->kpca_components().set_size(
    arguments_in.num_kpca_components_in_, product.n_cols);
  for(unsigned int j = 0; j < product.n_cols; j++) {
    for(int i = 0; i < arguments_in.num_kpca_components_in_; i++) {
      result_out->kpca_components().at(i, j) =
        product.at(i, j) / global_lengths[i];
    }
  }
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::ComputeEigenDecomposition_(
  double num_standard_deviations,
  int num_reference_samples,
  const mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > &arguments_in,
  mlpack::distributed_kpca::KpcaResult *result_out) {

  // The master generates a set of random Fourier features and do
  // a broadcast.
  arma::mat random_variates;
  boost::scoped_array<arma::vec> random_variate_aliases(
    new arma::vec[num_random_fourier_features_eigen_]);
  GenerateRandomFourierFeatures_(
    num_random_fourier_features_eigen_,
    &random_variates, random_variate_aliases);

  // Each process computes its local mean, and performs an
  // all-reduction, if we are performing a centered version of KPCA.
  core::monte_carlo::MeanVariancePairMatrix local_mean;
  core::monte_carlo::MeanVariancePairMatrix global_mean;
  if(arguments_in.do_centering_) {
    mlpack::series_expansion::RandomFeature<TableType>::
    ThreadedNormalizedAverageTransform(
      arguments_in.num_threads_in_,
      *(arguments_in.reference_table_->local_table()),
      random_variate_aliases, num_random_fourier_features_eigen_, &local_mean);
    boost::mpi::all_reduce(
      *world_, local_mean, global_mean,
      core::parallel::CombineMeanVariancePairMatrix());
  }

  // Each process computes its local covariance and the master
  // reduces them to form the global covariance.
  core::monte_carlo::MeanVariancePairMatrix local_covariance;
  core::monte_carlo::MeanVariancePairMatrix global_covariance;
  mlpack::series_expansion::RandomFeature<TableType>::
  ThreadedCovarianceTransform(
    arguments_in.num_threads_in_,
    *(arguments_in.reference_table_->local_table()), arguments_in.do_centering_,
    global_mean, random_variate_aliases, num_random_fourier_features_eigen_,
    &local_covariance, &result_out->reference_projections());
  boost::mpi::reduce(
    *world_, local_covariance, global_covariance,
    core::parallel::CombineMeanVariancePairMatrix(), 0);

  // The master eigendecomposes the converged global covariance and
  // does a broadcast.
  if(world_->rank() == 0) {
    arma::mat global_covariance_alias;
    arma::vec kernel_eigenvalues;
    arma::mat covariance_eigenvectors;
    global_covariance.sample_means(&global_covariance_alias);
    arma::eig_sym(
      kernel_eigenvalues, covariance_eigenvectors, global_covariance_alias);
    result_out->set_eigendecomposition_results(
      kernel_eigenvalues, covariance_eigenvectors);
  }
  boost::mpi::broadcast(
    *world_, result_out->kernel_eigenvalues(), 0);
  boost::mpi::broadcast(*world_, result_out->covariance_eigenvectors(), 0);
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::NaiveKernelEigenvectors_(
  int num_kpca_components_in_,
  bool do_centering,
  DistributedTableType *reference_table_in,
  const arma::mat &kpca_components) {

  printf("Verifying naively\n");

  arma::mat naive_kernel_matrix;
  naive_kernel_matrix.set_size(
    reference_table_in->n_entries(), reference_table_in->n_entries());
  for(int j = 0; j < reference_table_in->n_entries(); j++) {
    arma::vec point_j;
    reference_table_in->local_table()->get(j, &point_j);
    for(int i = 0; i < reference_table_in->n_entries(); i++) {
      arma::vec point_i;
      reference_table_in->local_table()->get(i, &point_i);
      double squared_distance = metric_.DistanceSq(point_i, point_j);
      double kernel_value = kernel_.EvalUnnormOnSq(squared_distance);
      naive_kernel_matrix.at(i, j) =
        kernel_value /
        static_cast<double>(reference_table_in->n_entries());
    }
  }
  core::monte_carlo::MeanVariancePairVector kernel_averages;
  kernel_averages.Init(reference_table_in->n_entries());
  core::monte_carlo::MeanVariancePair overall_average;
  if(do_centering) {
    for(int j = 0; j < reference_table_in->n_entries(); j++) {
      for(int i = 0; i < reference_table_in->n_entries(); i++) {
        kernel_averages[j].push_back(naive_kernel_matrix.at(i, j));
        overall_average.push_back(naive_kernel_matrix.at(i, j));
      }
    }
    for(unsigned int j = 0; j < naive_kernel_matrix.n_cols; j++) {
      for(unsigned int i = 0; i < naive_kernel_matrix.n_rows; i++) {
        naive_kernel_matrix.at(i, j) = naive_kernel_matrix.at(i, j) -
                                       kernel_averages[j].sample_mean() -
                                       kernel_averages[i].sample_mean() +
                                       overall_average.sample_mean();
      }
    }
  }

  // Eigendecompose the naive kernel matrix.
  arma::vec naive_kernel_eigenvalues;
  arma::mat naive_kernel_eigenvectors;
  printf("Calling on naive!\n");
  arma::eig_sym(
    naive_kernel_eigenvalues, naive_kernel_eigenvectors, naive_kernel_matrix);

  // Compare against the random Gram matrix generated from the random
  // Fourier features.
  arma::mat random_matrix;
  random_matrix.set_size(
    reference_table_in->n_entries(), reference_table_in->n_entries());
  boost::scoped_array<arma::vec> transformed_points(
    new arma::vec[reference_table_in->n_entries()]);

  // The master generates a set of random Fourier features and do a
  // broadcast.
  const int num_random_fourier_features = 20;
  arma::mat random_variates;
  boost::scoped_array<arma::vec> random_variate_aliases(
    new arma::vec[num_random_fourier_features]);
  GenerateRandomFourierFeatures_(
    num_random_fourier_features, &random_variates, random_variate_aliases);
  arma::vec average_fourier_features;
  average_fourier_features.zeros(num_random_fourier_features * 2);
  for(int j = 0; j < reference_table_in->n_entries(); j++) {
    arma::vec original_point;
    reference_table_in->local_table()->get(j, &original_point);
    mlpack::series_expansion::RandomFeature<TableType>::Transform(
      original_point, random_variate_aliases,
      num_random_fourier_features, & (transformed_points[j]));
    double first_factor = static_cast<double>(j) / static_cast<double>(j + 1);
    double second_factor = 1.0 - first_factor;
    average_fourier_features =
      first_factor * average_fourier_features +
      second_factor * transformed_points[j];
  }

  // Do centering if requested.
  if(do_centering) {
    for(int j = 0; j < reference_table_in->n_entries(); j++) {
      transformed_points[j] -= average_fourier_features;
    }
  }
  for(int j = 0; j < reference_table_in->n_entries(); j++) {
    for(int i = 0; i < reference_table_in->n_entries(); i++) {
      double kernel_value =
        arma::dot(transformed_points[i], transformed_points[j]);
      random_matrix.at(i, j) =
        kernel_value /
        static_cast<double>(
          reference_table_in->n_entries() * num_random_fourier_features);
    }
  }
  arma::vec random_matrix_eigenvalues;
  arma::mat random_matrix_eigenvectors;
  printf("Calling on Fourier features.\n");
  arma::eig_sym(
    random_matrix_eigenvalues, random_matrix_eigenvectors, random_matrix);

  // Covariance matrix.
  arma::mat random_matrix_covariance;
  random_matrix_covariance.zeros(
    2 * num_random_fourier_features, 2 * num_random_fourier_features);
  arma::mat transformed_points_mat;
  transformed_points_mat.set_size(
    2 * num_random_fourier_features, reference_table_in->n_entries());
  for(int k = 0; k < reference_table_in->n_entries(); k++) {
    for(int j = 0; j < num_random_fourier_features * 2; j++) {
      transformed_points_mat.at(j, k) = transformed_points[k][j] /
                                        sqrt(num_random_fourier_features) ;
      for(int i = 0; i < num_random_fourier_features * 2; i++) {
        random_matrix_covariance.at(i, j) +=
          transformed_points[k][i] * transformed_points[k][j] /
          static_cast<double>(num_random_fourier_features);
      }
    }
  }
  random_matrix_covariance *=
    (1.0 / static_cast<double>(reference_table_in->n_entries()));
  arma::vec covariance_eigenvalues;
  arma::mat covariance_eigenvectors;
  printf("Calling on covariance:\n");
  arma::eig_sym(
    covariance_eigenvalues, covariance_eigenvectors, random_matrix_covariance);

  printf("\nNaive kernel matrix eigenvalues:\n");
  naive_kernel_eigenvalues.print();
  printf("\nRandom Gram matrix eigenvalues:\n");
  random_matrix_eigenvalues.print();
  arma::mat product = arma::trans(covariance_eigenvectors) *
                      transformed_points_mat;
  for(unsigned int i = 0; i < product.n_rows; i++) {
    double magnitude = 0.0;
    for(unsigned int j = 0; j < transformed_points_mat.n_cols; j++) {
      magnitude += core::math::Sqr(product.at(i, j));
    }
    printf("Checking %g\n", magnitude / static_cast<double>(product.n_cols));
  }
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::NaiveWeightedKernelAverage_(
  bool do_centering,
  double relative_error_in,
  double absolute_error_in,
  DistributedTableType *reference_table_in,
  DistributedTableType *query_table_in,
  const arma::mat &weights,
  const core::monte_carlo::MeanVariancePairMatrix
  &approx_weighted_kernel_averages) {

  // Compute the kernel averages.
  core::monte_carlo::MeanVariancePairVector naive_query_kernel_averages;
  core::monte_carlo::MeanVariancePair global_reference_kernel_average;
  core::monte_carlo::MeanVariancePairVector naive_reference_kernel_averages;
  naive_query_kernel_averages.Init(query_table_in->n_entries());
  naive_reference_kernel_averages.Init(reference_table_in->n_entries());

  if(do_centering) {
    for(int i = 0; i < query_table_in->n_entries(); i++) {

      // The query point.
      arma::vec query_point;
      query_table_in->local_table()->get(i, &query_point);
      for(int j = 0; j < reference_table_in->n_entries(); j++) {

        // The reference point.
        arma::vec reference_point;
        reference_table_in->local_table()->get(j, &reference_point);

        // Compute the squared distance and the kernel value.
        double squared_distance =
          metric_.DistanceSq(query_point, reference_point);
        double kernel_value =
          kernel_.EvalUnnormOnSq(squared_distance);
        naive_query_kernel_averages[i].push_back(kernel_value);
      }
    }
    if(query_table_in != reference_table_in) {
      for(int i = 0; i < reference_table_in->n_entries(); i++) {

        // The outer reference point.
        arma::vec outer_reference_point;
        reference_table_in->local_table()->get(i, &outer_reference_point);
        for(int j = 0; j < reference_table_in->n_entries(); j++) {

          // The reference point.
          arma::vec inner_reference_point;
          reference_table_in->local_table()->get(j, &inner_reference_point);

          // Compute the squared distance and the kernel value.
          double squared_distance =
            metric_.DistanceSq(outer_reference_point, inner_reference_point);
          double kernel_value =
            kernel_.EvalUnnormOnSq(squared_distance);
          naive_reference_kernel_averages[i].push_back(kernel_value);
        }
      }
    }
    else {
      naive_reference_kernel_averages.CopyValues(naive_query_kernel_averages);
    }
    for(int i = 0; i < reference_table_in->n_entries(); i++) {
      global_reference_kernel_average.push_back(
        naive_reference_kernel_averages[i].sample_mean());
    }

    printf("References:\n");
    for(int i = 0; i < reference_table_in->n_entries(); i++) {
      printf("%g ", naive_reference_kernel_averages[i].sample_mean());
    }
    printf("\n");
    printf("Global average: %g\n", global_reference_kernel_average.sample_mean());
  }

  // Allocate the weighted kernel sum slots.
  core::monte_carlo::MeanVariancePairMatrix naive_weighted_kernel_averages;
  naive_weighted_kernel_averages.Init(
    weights.n_rows, query_table_in->n_entries());

  for(int i = 0; i < query_table_in->n_entries(); i++) {

    // The query point.
    arma::vec query_point;
    query_table_in->local_table()->get(i, &query_point);

    for(int j = 0; j < reference_table_in->n_entries(); j++) {

      // The reference point.
      arma::vec reference_point;
      reference_table_in->local_table()->get(j, &reference_point);

      // Compute the squared distance and the kernel value.
      double squared_distance =
        metric_.DistanceSq(query_point, reference_point);
      double kernel_value =
        kernel_.EvalUnnormOnSq(squared_distance);
      if(do_centering) {
        kernel_value = kernel_value -
                       naive_query_kernel_averages[i].sample_mean() -
                       naive_reference_kernel_averages[j].sample_mean() +
                       global_reference_kernel_average.sample_mean();
      }
      for(unsigned int k = 0; k < weights.n_rows; k++) {
        naive_weighted_kernel_averages.get(k, i).push_back(
          weights.at(k, j) * kernel_value);
      }
    }
  }

  // Extract the means of both naive and approximated quantities.
  arma::mat approximated;
  arma::mat naive;
  naive_weighted_kernel_averages.sample_means(&naive);
  approx_weighted_kernel_averages.sample_means(&approximated);
  int error_bound_satisifed_count = 0;
  for(unsigned int j = 0; j < naive.n_cols; j++) {
    double naive_l1_norm = 0.0;
    double approx_l1_norm = 0.0;
    double error_l1_norm = 0.0;
    for(unsigned int i = 0; i < naive.n_rows; i++) {
      naive_l1_norm += fabs(naive.at(i, j));
      approx_l1_norm += fabs(approximated.at(i, j));
      error_l1_norm += fabs(naive.at(i, j) - approximated.at(i, j));
      printf(
        "%g vs %g: %g %g\n", approximated.at(i, j), naive.at(i, j),
        error_l1_norm, relative_error_in * naive_l1_norm + absolute_error_in);
    }
    if(error_l1_norm <= relative_error_in * naive_l1_norm + absolute_error_in) {
      error_bound_satisifed_count++;
    }
  }
  printf("%d averages satisified the bound.\n", error_bound_satisifed_count);
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::ComputeWeightedKernelAverage_(
  int num_threads_in,
  double relative_error_in,
  double absolute_error_in,
  double num_standard_deviations,
  int num_reference_samples,
  int num_random_fourier_features,
  DistributedTableType *reference_table_in,
  DistributedTableType *query_table_in,
  const arma::mat &weights,
  bool do_centering_in,
  core::monte_carlo::MeanVariancePairMatrix *kernel_sums) {

  // Allocate the weighted kernel sum slot.
  kernel_sums->Init(weights.n_rows, query_table_in->n_entries());

  // Indicates the convergence of each kernel sum.
  core::monte_carlo::MeanVariancePairVector l1_norm_history;
  l1_norm_history.Init(query_table_in->n_entries());
  std::deque< bool > converged(query_table_in->n_entries(), false);

  // Call the computation.
  int num_iterations = 0;
  bool all_query_converged = true;

  // Temporary vector used for generating a random combination.
  std::vector<int> random_combination;
  boost::scoped_array<arma::vec> random_variate_aliases(
    new arma::vec[num_random_fourier_features]);
  do {

    // The master generates a set of random Fourier features and do a
    // broadcast.
    arma::mat random_variates;
    GenerateRandomFourierFeatures_(
      num_random_fourier_features, &random_variates, random_variate_aliases);

    // Each process computes the local feature map average and the
    // global process combines it.
    core::monte_carlo::MeanVariancePairMatrix local_mean;
    core::monte_carlo::MeanVariancePairMatrix global_mean;

    // Depending on whether the centering is requested or not, compute
    // the center of the training set in the feature space.
    if(do_centering_in) {
      mlpack::series_expansion::RandomFeature<TableType>::
      ThreadedNormalizedAverageTransform(
        num_threads_in,
        *(reference_table_in->local_table()),
        random_variate_aliases, num_random_fourier_features_eigen_,
        &local_mean);
      boost::mpi::all_reduce(
        *world_, local_mean, global_mean,
        core::parallel::CombineMeanVariancePairMatrix());

      // Multiply back by this factor to make sure that the random
      // features for the training set are centered properly.
      global_mean.scale(sqrt(num_random_fourier_features_eigen_));
    }
    else {

      // Zero global mean for non-centered KPCA.
      global_mean.Init(1, random_variates.n_cols * 2);
    }

    // Each process computes the sum of the projections of the local
    // reference set and does an all-reduction to compute the global
    // sum projections.
    core::monte_carlo::MeanVariancePairMatrix local_reference_average;
    mlpack::series_expansion::RandomFeature <
    TableType >::WeightedAverageTransform(
      *(reference_table_in->local_table()),
      weights, num_reference_samples,
      random_variate_aliases, num_random_fourier_features, global_mean,
      &random_combination, &local_reference_average);
    core::monte_carlo::MeanVariancePairMatrix global_reference_average;
    boost::mpi::all_reduce(
      *world_, local_reference_average, global_reference_average,
      core::parallel::CombineMeanVariancePairMatrix());

    // Each process computes the local projection of each query point
    // and adds up.
    bool all_local_query_converged =
      mlpack::distributed_kpca::
      ThreadedConvergence<DistributedTableType>::ThreadedCheck(
        num_threads_in,
        relative_error_in,
        absolute_error_in,
        num_standard_deviations,
        num_reference_samples,
        reference_table_in,
        query_table_in,
        kernel_sums,
        converged,
        random_variate_aliases,
        num_random_fourier_features,
        l1_norm_history,
        num_iterations,
        max_num_iterations_,
        &global_reference_average);

    // Increment the number of iterations.
    num_iterations++;

    // Do an all-reduction to find out we are all done.
    boost::mpi::all_reduce(
      *world_, all_local_query_converged,
      all_query_converged, std::logical_and<bool>());

  } // Terminate the loop only if all processes are done.
  while(! all_query_converged);

  printf("Process %d converged in %d iterations...\n", world_->rank(),
         num_iterations);

  // Do a post-corrrection to the number of terms for each Monte Carlo
  // estimate.
  for(int j = 0; j < kernel_sums->n_cols(); j++) {
    for(int i = 0; i < kernel_sums->n_rows(); i++) {
      kernel_sums->get(i, j).set_total_num_terms(
        effective_num_reference_points_);
    }
  }
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca<DistributedTableType, KernelType>::Compute(
  const mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > &arguments_in,
  mlpack::distributed_kpca::KpcaResult *result_out) {

  // The number of Fourier features to sample in each round.
  int num_random_fourier_features = num_random_fourier_features_eigen_;

  // The number of reference points to pick in each round.
  int num_reference_samples =
    std::min(1000, arguments_in.reference_table_->n_entries());

  // Determine the number of standard deviation coverage.
  double cumulative_probability = arguments_in.probability_ +
                                  0.5 * (1.0 - arguments_in.probability_);
  double num_standard_deviations =
    (cumulative_probability > 0.999) ?
    3.0 : boost::math::quantile(normal_dist_, cumulative_probability);

  // Barrier so that every process is here.
  world_->barrier();

  // The MPI timer.
  boost::mpi::timer timer;

  // If the mode is KDE, then we need to compute the kernel sum
  // between the query set and the reference set.
  core::monte_carlo::MeanVariancePairMatrix query_kernel_averages;
  if(arguments_in.mode_ == "kde") {
    ComputeWeightedKernelAverage_(
      arguments_in.num_threads_in_,
      arguments_in.relative_error_,
      arguments_in.absolute_error_,
      num_standard_deviations,
      num_reference_samples,
      num_random_fourier_features,
      arguments_in.reference_table_,
      arguments_in.query_table_,
      arguments_in.reference_table_->local_table()->weights(),
      false,
      &query_kernel_averages);

    if(arguments_in.do_naive_) {
      NaiveWeightedKernelAverage_(
        false,
        arguments_in.relative_error_,
        arguments_in.absolute_error_,
        arguments_in.reference_table_,
        arguments_in.query_table_,
        arguments_in.reference_table_->local_table()->weights(),
        query_kernel_averages);
    }
  }

  // If the mode is KPCA, then need to compute the eigenvectors.
  if(arguments_in.mode_ == "kpca") {
    if(world_->rank() == 0) {
      std::cout << "Starting to compute eigendecomposition...\n";
    }
    ComputeEigenDecomposition_(
      num_standard_deviations, num_reference_samples,
      arguments_in, result_out);
  }

  // Allocate the space for the results.
  result_out->Init(
    arguments_in.num_kpca_components_in_,
    arguments_in.reference_table_->n_entries(),
    arguments_in.query_table_->n_entries());

  // If the mode is KPCA, then need to finalize the kernel
  // eigenvectors.
  if(arguments_in.mode_ == "kpca") {
    if(world_->rank() == 0) {
      std::cout << "Starting to finalize the eigendecomposition...\n";
    }
    FinalizeKernelEigenvectors_(
      num_standard_deviations, num_reference_samples,
      arguments_in, result_out);
    if(world_->rank() == 0) {
      printf(
        "Spent up to %g seconds in kernel eigenvectors.\n", timer.elapsed());
    }

    if(arguments_in.do_naive_) {
      NaiveKernelEigenvectors_(
        arguments_in.num_kpca_components_in_,
        arguments_in.do_centering_,
        arguments_in.reference_table_,
        result_out->kpca_components());
    }
  }
  else {

    // Otherwise set everything to one.
    result_out->kpca_components().set_size(
      1, arguments_in.reference_table_->n_entries());
    result_out->kpca_components().fill(1.0);
  }

  // If the mode is KPCA, then we need to compute the projection of
  // each query point onto the KPCA components.
  if(arguments_in.mode_ == "kpca") {

    if(world_->rank() == 0) {
      std::cout << "Starting the projection step...\n";
    }

    core::monte_carlo::MeanVariancePairMatrix query_kpca_projections;
    ComputeWeightedKernelAverage_(
      arguments_in.num_threads_in_,
      arguments_in.relative_error_,
      arguments_in.absolute_error_,
      num_standard_deviations,
      num_reference_samples,
      num_random_fourier_features,
      arguments_in.reference_table_,
      arguments_in.query_table_,
      result_out->kpca_components(),
      arguments_in.do_centering_,
      &query_kpca_projections);

    if(arguments_in.do_naive_) {

      // Check again on the centered result, if naive computation is
      // required.
      NaiveWeightedKernelAverage_(
        arguments_in.do_centering_,
        arguments_in.relative_error_,
        arguments_in.absolute_error_,
        arguments_in.reference_table_,
        arguments_in.query_table_,
        result_out->kpca_components(),
        query_kpca_projections);
    }

    // Export the results after scaling it.
    query_kpca_projections.scale(effective_num_reference_points_);
    result_out->Export(
      num_standard_deviations, mult_const_,
      correction_term_, query_kpca_projections);
  }
  else {

    // Export the KDE result.
    result_out->Export(
      num_standard_deviations, mult_const_,
      correction_term_, query_kernel_averages);
  }

  // Barrier so that every process is done.
  world_->barrier();
  if(world_->rank() == 0) {
    printf(
      "Spent %g seconds in computation.\n", timer.elapsed());
  }
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca<DistributedTableType, KernelType>::Init(
  boost::mpi::communicator &world_in,
  mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > &arguments_in) {

  // Set the number of threads.
  omp_set_num_threads(arguments_in.num_threads_in_);

  // Set the maximum number of iterations.
  max_num_iterations_ = arguments_in.max_num_iterations_in_;

  // Initialize the kernel.
  kernel_.Init(arguments_in.bandwidth_);
  mult_const_ =
    (arguments_in.mode_ == "kpca") ?
    1.0 :
    1.0 /
    kernel_.CalcNormConstant(
      arguments_in.reference_table_->n_attributes());

  // Initialize the dimension.
  num_dimensions_ = arguments_in.reference_table_->n_attributes();

  // Set the communicator.
  world_ = &world_in;

  // This case implies that the query equals the reference.
  if(arguments_in.query_table_ == NULL) {
    arguments_in.query_table_ = arguments_in.reference_table_;
  }

  // Initialize the multiplicative constant.
  int total_sum = 0;
  for(int i = 0; i < world_->size(); i++) {
    total_sum += arguments_in.reference_table_->local_n_entries(i);
  }
  effective_num_reference_points_ =
    (arguments_in.reference_table_ == arguments_in.query_table_ &&
     arguments_in.mode_ == "kde") ?
    (total_sum - 1) : total_sum;

  // In case the mode is KDE, and is monochromatic.
  correction_term_ =
    (arguments_in.mode_ == "kde") ?
    1.0 / static_cast<double>(
      (arguments_in.reference_table_ == arguments_in.query_table_) ?
      (effective_num_reference_points_ + 1) :
      effective_num_reference_points_) : 0;

  // The number of Fourier features sampled for eigendecomposition.
  num_random_fourier_features_eigen_ =
    (arguments_in.mode_ == "kpca") ?
    arguments_in.num_kpca_components_in_ + 5 :
    std::min(20, arguments_in.reference_table_->n_attributes() + 5) ;
}
}
}

#endif
