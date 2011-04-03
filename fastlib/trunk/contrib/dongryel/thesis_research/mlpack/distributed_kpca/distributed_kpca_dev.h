/** @file distributed_kpca_dev.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_DEV_H
#define MLPACK_DISTRIBUTED_KPCA_DISTRIBUTED_KPCA_DEV_H

#include "core/gnp/distributed_dualtree_dfs_dev.h"
#include "core/metric_kernels/lmetric.h"
#include "core/monte_carlo/mean_variance_pair_matrix.h"
#include "core/table/memory_mapped_file.h"
#include "core/table/transform.h"
#include "mlpack/distributed_kpca/distributed_kpca.h"
#include "mlpack/series_expansion/random_feature.h"

namespace core {
namespace parallel {

class AddDensePoint:
  public std::binary_function <
  core::table::DensePoint,
  core::table::DensePoint,
    core::table::DensePoint > {

  public:
    const core::table::DensePoint operator()(
      const core::table::DensePoint &a,
      const core::table::DensePoint &b) const {

      core::table::DensePoint sum;
      sum.Copy(a);
      arma::vec sum_alias;
      core::table::DensePointToArmaVec(sum, &sum_alias);
      arma::vec b_alias;
      core::table::DensePointToArmaVec(b, &b_alias);
      sum_alias += b_alias;
      return sum;
    }
};

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

template<>
class is_commutative <
  core::parallel::AddDensePoint,
  core::table::DensePoint  > :
  public boost::mpl::true_ {

};

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
extern core::table::MemoryMappedFile *global_m_file_;
}
}

namespace mlpack {
namespace distributed_kpca {

template<typename DistributedTableType, typename KernelType>
DistributedKpca<DistributedTableType, KernelType>::DistributedKpca() {
  world_ = NULL;
  mult_const_ = 0.0;
  effective_num_reference_points_ = 0.0;
  correction_term_ = 0.0;
  num_random_fourier_features_eigen_ = 1;
  num_dimensions_ = 0;
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::GenerateRandomFourierFeatures_(
  int num_random_fourier_features,
  std::vector <
  core::table::DensePoint > *random_variates,
  std::vector< arma::vec > *random_variate_aliases) {

  random_variates->resize(num_random_fourier_features);
  random_variate_aliases->resize(num_random_fourier_features);
  if(world_->rank() == 0) {
    for(int i = 0; i < num_random_fourier_features; i++) {

      // Draw a random Fourier feature.
      kernel_.DrawRandomVariate(
        num_dimensions_, & (*random_variates)[i]);
    }
  }
  boost::mpi::broadcast(*world_, *random_variates, 0);
  for(int i = 0; i < num_random_fourier_features; i++) {
    core::table::DensePointToArmaVec(
      (*random_variates)[i], &((*random_variate_aliases)[i]));
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

  core::monte_carlo::MeanVariancePairMatrix local_kpca_components;
  local_kpca_components.Init(
    arguments_in.num_kpca_components_in_,
    arguments_in.reference_table_->n_entries());

  // The main loop.
  int num_iterations = 0;
  do {

    // The master generates the set of random Fourier features.
    std::vector <
    core::table::DensePoint > random_variates;
    std::vector< arma::vec > random_variate_aliases;
    GenerateRandomFourierFeatures_(
      num_random_fourier_features_eigen_,
      &random_variates, &random_variate_aliases);

    // Each process independently computes its own portion of kernel
    // eigenvectors.
    mlpack::series_expansion::RandomFeature::AccumulateRotationTransform(
      *(arguments_in.reference_table_->local_table()),
      result_out->covariance_eigenvectors(),
      random_variate_aliases, &local_kpca_components);

    // Each process determines whether the kernel eigenvector estimates
    // are good enough, and notifies the master.
    bool all_components_converged = true;
    for(int i = 0; all_components_converged &&
        i < local_kpca_components.n_rows(); i++) {
      double left_hand_side = 0.0;
      double right_hand_side = 0.0;
      for(int j = 0; j < local_kpca_components.n_cols(); j++) {
        left_hand_side +=
          core::math::Sqr(num_standard_deviations) *
          local_kpca_components.get(i, j).sample_mean_variance();
        right_hand_side +=
          core::math::Sqr(local_kpca_components.get(i, j).sample_mean());
      }
      all_components_converged =
        (left_hand_side <= arguments_in.relative_error_ * right_hand_side +
         arguments_in.absolute_error_);
    }

    // Quit after 10 iterations.
    all_components_converged = (num_iterations >= 10);

    bool all_done = true;
    boost::mpi::all_reduce(
      *world_, all_components_converged, all_done, std::logical_and<bool>());
    if(all_done) {
      break;
    }
    num_iterations++;
  }
  while(true);

  if(world_->rank() == 0) {
    std::cout << "KPCA eigenvector finalizing took " << num_iterations <<
              " iterations.\n";
  }

  // Extract.
  local_kpca_components.sample_means(& (result_out->kpca_components()));

  // Normalize each KPCA components. Each process needs to compute its
  // squared length and participate in the global all-reduce step.
  core::table::DensePoint local_squared_length_contributions;
  local_squared_length_contributions.Init(
    result_out->kpca_components().n_rows());
  local_squared_length_contributions.SetZero();
  core::table::DensePoint global_lengths;
  for(int j = 0; j < result_out->kpca_components().n_cols(); j++) {
    for(int i = 0; i < result_out->kpca_components().n_rows(); i++) {
      local_squared_length_contributions[i] +=
        core::math::Sqr(result_out->kpca_components().get(i, j));
    }
  }
  boost::mpi::all_reduce(
    *world_, local_squared_length_contributions,
    global_lengths, core::parallel::AddDensePoint());

  // Take the square roots of each KPCA component to compute the
  // actual normalization factor.
  for(int i = 0; i < global_lengths.length(); i++) {
    global_lengths[i] = sqrt(global_lengths[i]);
  }
  for(int j = 0; j < result_out->kpca_components().n_cols(); j++) {
    for(int i = 0; i < result_out->kpca_components().n_rows(); i++) {
      result_out->kpca_components().set(
        i, j, result_out->kpca_components().get(i, j) / global_lengths[i]);
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

  // The master computes the eigenvectors.
  core::monte_carlo::MeanVariancePairMatrix global_covariance;
  global_covariance.Init(
    2 * num_random_fourier_features_eigen_,
    2 * num_random_fourier_features_eigen_);

  int num_iterations = 0;
  do {

    // The master generates a set of random Fourier features and do
    // a broadcast.
    std::vector <
    core::table::DensePoint > random_variates;
    std::vector< arma::vec > random_variate_aliases;
    GenerateRandomFourierFeatures_(
      num_random_fourier_features_eigen_,
      &random_variates, &random_variate_aliases);

    // Each process computes its local covariance and the master
    // reduces them to form the global covariance.
    core::monte_carlo::MeanVariancePairMatrix local_covariance;
    core::monte_carlo::MeanVariancePairMatrix global_covariance_in_this_round;
    mlpack::series_expansion::RandomFeature::CovarianceTransform(
      *(arguments_in.reference_table_->local_table()),
      num_reference_samples, random_variate_aliases,
      &local_covariance);
    boost::mpi::reduce(
      *world_, local_covariance, global_covariance_in_this_round,
      core::parallel::CombineMeanVariancePairMatrix(), 0);

    // The master determines whether the covariance estimates are
    // good enough.
    bool all_components_converged = true;
    if(world_->rank() == 0) {
      global_covariance.CombineWith(global_covariance_in_this_round);

      double total_frobenius_norm = 0.0;
      double total_error = 0.0;
      for(int j = 0; j < global_covariance.n_cols(); j++) {
        for(int i = 0; i <= j; i++) {
          total_frobenius_norm +=
            core::math::Sqr(global_covariance.get(i, j).sample_mean());
          total_error += core::math::Sqr(num_standard_deviations) *
                         global_covariance.get(i, j).sample_mean_variance();
        }
      }
      all_components_converged =
        (total_error <= arguments_in.relative_error_ *
         total_frobenius_norm + arguments_in.absolute_error_);
    }
    bool all_done = true;
    boost::mpi::all_reduce(
      *world_, all_components_converged, all_done, std::logical_and<bool>());

    num_iterations++;
    if(all_done) {
      break;
    }
  }
  while(true);

  if(world_->rank() == 0) {
    std::cout << "KPCA eigenvector preprocessing took " << num_iterations <<
              " iterations.\n";
  }

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
      arguments_in.num_kpca_components_in_,
      kernel_eigenvalues, covariance_eigenvectors);
  }
  boost::mpi::broadcast(
    *world_, result_out->kernel_eigenvalues(), 0);
  boost::mpi::broadcast(*world_, result_out->covariance_eigenvectors(), 0);
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca<DistributedTableType, KernelType>::ComputeKernelSum_(
  double relative_error_in,
  double absolute_error_in,
  double num_standard_deviations,
  int num_reference_samples,
  int num_random_fourier_features,
  DistributedTableType *reference_table_in,
  DistributedTableType *query_table_in,
  const core::table::DenseMatrix &weights,
  core::monte_carlo::MeanVariancePairMatrix *kernel_sums) {

  // Allocate the weighted kernel sum slot.
  kernel_sums->Init(weights.n_rows(), query_table_in->n_entries());

  // Indicates the convergence of each kernel sum.
  core::monte_carlo::MeanVariancePairVector frobenius_norm_history;
  frobenius_norm_history.Init(query_table_in->n_entries());
  std::vector< bool > converged(query_table_in->n_entries(), false);

  // Call the computation.
  int num_iterations = 0;
  bool all_query_converged = true;
  do {

    // The master generates a set of random Fourier features and do a
    // broadcast.
    std::vector <
    core::table::DensePoint > random_variates(num_random_fourier_features);
    std::vector< arma::vec > random_variate_aliases(
      num_random_fourier_features);
    GenerateRandomFourierFeatures_(
      num_random_fourier_features, &random_variates, &random_variate_aliases);

    // Each process computes the sum of the projections of the local
    // reference set and does an all-reduction to compute the global
    // sum projections.
    core::monte_carlo::MeanVariancePairMatrix local_reference_average;
    mlpack::series_expansion::RandomFeature::AverageTransform(
      *(reference_table_in->local_table()),
      weights, num_reference_samples,
      random_variate_aliases, &local_reference_average);
    core::monte_carlo::MeanVariancePairMatrix global_reference_average;
    boost::mpi::all_reduce(
      *world_, local_reference_average, global_reference_average,
      core::parallel::CombineMeanVariancePairMatrix());

    // Each process computes the local projection of each query point
    // and adds up.
    bool all_local_query_converged = true;
    for(int i = 0; i < query_table_in->local_table()->n_entries(); i++) {

      if(converged[i]) {

        // If already converged, skip.
        continue;
      }

      arma::vec query_point;
      query_table_in->local_table()->get(i, &query_point);
      arma::vec query_point_projected;
      mlpack::series_expansion::RandomFeature::Transform(
        query_point, random_variate_aliases, &query_point_projected);

      double frobenius_norm = 0.0;
      for(int k = 0; k < weights.n_rows(); k++) {
        for(int j = 0; j < num_random_fourier_features; j++) {

          // You need to multiply by the factor of two since Fourier
          // features come in pairs of cosine and sines.
          kernel_sums->get(k, i).ScaledCombineWith(
            2.0 * query_point_projected[j],
            global_reference_average.get(k, j));
          kernel_sums->get(k, i).ScaledCombineWith(
            2.0 * query_point_projected[j + num_random_fourier_features],
            global_reference_average.get(k, j + num_random_fourier_features));
        }

        // Add up the frobenius norm contribution.
        frobenius_norm +=
          core::math::Sqr(kernel_sums->get(k, i).sample_mean());

      } // end of checking the given KPCA component.

      // Add to the history.
      frobenius_norm_history[i].push_back(frobenius_norm);

      // Start checking the convergence after 5 iterations.
      if(num_iterations > 5) {
        converged[i] = (
                         num_standard_deviations *
                         sqrt(
                           frobenius_norm_history[i].sample_mean_variance()) <=
                         relative_error_in *
                         frobenius_norm_history[i].sample_mean() +
                         absolute_error_in);
      }
      all_local_query_converged = converged[i];

    } // end of looping over each local query.

    // Increment the number of iterations.
    num_iterations++;

    // Do an all-reduction to find out we are all done.
    boost::mpi::all_reduce(
      *world_, all_local_query_converged,
      all_query_converged, std::logical_and<bool>());

  } // Terminate the loop only if all processes are done.
  while(! all_query_converged);
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca <
DistributedTableType, KernelType >::PostProcessKpcaProjections_(
  const core::monte_carlo::MeanVariancePairMatrix &reference_kernel_sums,
  const core::monte_carlo::MeanVariancePairMatrix &query_kernel_sums,
  const core::monte_carlo::MeanVariancePair &average_reference_kernel_sum,
  const core::table::DenseMatrix &kpca_components,
  core::monte_carlo::MeanVariancePairMatrix *query_kpca_projections) {

  // The reference dot products (need to be computed across all
  // processes). The last trailing components are the sum of the KPCA
  // components across all processes.
  core::monte_carlo::MeanVariancePairMatrix local_reference_dot_products;
  local_reference_dot_products.Init(1, 2 * query_kpca_projections->n_rows());
  local_reference_dot_products.set_total_num_terms(
    reference_kernel_sums.n_cols());
  core::monte_carlo::MeanVariancePairMatrix global_reference_dot_products;

  // For each KPCA component, each process computes its local portion
  // of the dot products.
  for(int j = 0; j < reference_kernel_sums.n_cols(); j++) {
    for(int i = 0; i < query_kpca_projections->n_rows(); i++) {
      local_reference_dot_products.get(0, i).push_back(
        kpca_components.get(i, j) *
        reference_kernel_sums.get(0, j).sample_mean());
      local_reference_dot_products.get(
        0, i + query_kpca_projections->n_rows()).push_back(
          kpca_components.get(i, j));
    }
  }
  boost::mpi::all_reduce(
    *world_, local_reference_dot_products, global_reference_dot_products,
    core::parallel::CombineMeanVariancePairMatrix());

  // Now do the correction.
  for(int j = 0; j < query_kpca_projections->n_cols(); j++) {
    for(int i = 0; i < query_kpca_projections->n_rows(); i++) {
      query_kpca_projections->get(i, j).ScaledCombineWith(
        -1.0, local_reference_dot_products.get(0, i));
      query_kpca_projections->get(i, j).ScaledCombineWith(
        - local_reference_dot_products.get(
          0, i + query_kpca_projections->n_rows()).sample_mean(),
        query_kernel_sums.get(0, j));
      query_kpca_projections->get(i, j).ScaledCombineWith(
        - local_reference_dot_products.get(
          0, i + query_kpca_projections->n_rows()).sample_mean(),
        average_reference_kernel_sum);
    }
  }
}

template<typename DistributedTableType, typename KernelType>
void DistributedKpca<DistributedTableType, KernelType>::Compute(
  const mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > &arguments_in,
  mlpack::distributed_kpca::KpcaResult *result_out) {

  // The number of Fourier features to sample in each round.
  const int num_random_fourier_features = 20;

  // The number of reference points to pick in each round.
  int num_reference_samples =
    std::min(1000, arguments_in.reference_table_->n_entries());

  // Determine the number of standard deviation coverage.
  double cumulative_probability = arguments_in.probability_ +
                                  0.5 * (1.0 - arguments_in.probability_);
  double num_standard_deviations =
    (cumulative_probability > 0.999) ?
    3.0 : boost::math::quantile(normal_dist_, cumulative_probability);
  printf("Number of standard deviation: %g\n", num_standard_deviations);

  // Barrier so that every process is here.
  world_->barrier();

  // The MPI timer.
  boost::mpi::timer timer;

  // If the mode is KPCA and the centering is requested or the mode is
  // KDE, then we need to compute the kernel sum between the query set
  // and the reference set.
  core::monte_carlo::MeanVariancePairMatrix query_kernel_sums;
  if((arguments_in.mode_ == "kpca" && arguments_in.do_centering_) ||
      arguments_in.mode_ == "kde") {
    ComputeKernelSum_(
      arguments_in.relative_error_,
      arguments_in.absolute_error_,
      num_standard_deviations,
      num_reference_samples,
      num_random_fourier_features,
      arguments_in.reference_table_,
      arguments_in.query_table_,
      arguments_in.reference_table_->local_table()->weights(),
      &query_kernel_sums);
  }

  // If the mode is KPCA and the centering is requested, and is not
  // monochromatic, then we need to compute the kernel sum between the
  // reference set and itself.
  core::monte_carlo::MeanVariancePairMatrix reference_kernel_sums;
  core::monte_carlo::MeanVariancePair average_reference_kernel_sum;
  if(arguments_in.mode_ == "kpca") {
    if(arguments_in.do_centering_ &&
        arguments_in.reference_table_ != arguments_in.query_table_) {
      ComputeKernelSum_(
        arguments_in.relative_error_,
        arguments_in.absolute_error_,
        num_standard_deviations,
        num_reference_samples,
        num_random_fourier_features,
        arguments_in.reference_table_,
        arguments_in.reference_table_,
        arguments_in.reference_table_->local_table()->weights(),
        &reference_kernel_sums);
    }
    else {
      reference_kernel_sums.Copy(query_kernel_sums);
    }

    // Set the total number of terms represented by the average
    // reference kernel sum, which is equal to the total number of
    // points in the global list of processes.
    average_reference_kernel_sum.set_total_num_terms(
      effective_num_reference_points_);
    for(int i = 0; i < reference_kernel_sums.n_cols(); i++) {
      average_reference_kernel_sum.push_back(
        reference_kernel_sums.get(0, i).sample_mean());
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
  }
  else {

    // Otherwise set everything to one.
    result_out->kpca_components().Init(
      1, arguments_in.reference_table_->n_entries());
    result_out->kpca_components().SetAll(1.0);
  }

  if(world_->rank() == 0) {
    std::cout << "Starting the projection step...\n";
  }

  // If the mode is KPCA, then we need to compute the projection of
  // each query point onto the KPCA components.
  if(arguments_in.mode_ == "kpca") {
    core::monte_carlo::MeanVariancePairMatrix query_kpca_projections;
    ComputeKernelSum_(
      arguments_in.relative_error_,
      arguments_in.absolute_error_,
      num_standard_deviations,
      num_reference_samples,
      num_random_fourier_features,
      arguments_in.reference_table_,
      arguments_in.query_table_,
      result_out->kpca_components(),
      &query_kpca_projections);

    // If the centering is requested, then we need a post-processing.
    if(arguments_in.do_centering_) {
      PostProcessKpcaProjections_(
        reference_kernel_sums, query_kernel_sums,
        average_reference_kernel_sum,
        result_out->kpca_components(),
        &query_kpca_projections);
    }

    // Export the results.
    result_out->Export(
      num_standard_deviations, mult_const_,
      correction_term_, query_kpca_projections);
  }
  else {

    // Export the KDE result.
    result_out->Export(
      num_standard_deviations, mult_const_,
      correction_term_, query_kernel_sums);
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
  double total_sum = 0;
  for(int i = 0; i < world_->size(); i++) {
    total_sum += arguments_in.reference_table_->local_n_entries(i);
  }
  effective_num_reference_points_ =
    (arguments_in.reference_table_ == arguments_in.query_table_ &&
     arguments_in.mode_ == "kde") ?
    (total_sum - 1.0) : total_sum;

  // In case the mode is KDE, and is monochromatic.
  correction_term_ =
    (arguments_in.mode_ == "kde") ?
    1.0 / static_cast<double>(
      (arguments_in.reference_table_ == arguments_in.query_table_) ?
      (effective_num_reference_points_ + 1.0) :
      effective_num_reference_points_) : 0.0;

  // The number of Fourier features sampled for eigendecomposition.
  num_random_fourier_features_eigen_ =
    arguments_in.num_kpca_components_in_ * 2;
}

bool DistributedKpcaArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  const std::vector<std::string> &args,
  boost::program_options::variables_map *vm) {

  boost::program_options::options_description desc("Available options");
  desc.add_options()(
    "help", "Print this information."
  )(
    "do_centering", "Apply centering to KPCA if present."
  )(
    "mode",
    boost::program_options::value<std::string>()->default_value("kde"),
    "OPTIONAL The algorithm mode. One of:"
    "  kde, kpca"
  )(
    "num_kpca_components_in",
    boost::program_options::value<int>()->default_value(3),
    "OPTIONAL The number of KPCA components to output."
  )(
    "kpca_components_out",
    boost::program_options::value<std::string>()->default_value(
      "kpca_components.csv"),
    "OPTIONAL output file for KPCA components."
  )(
    "references_in",
    boost::program_options::value<std::string>()->default_value(
      "random_dataset.csv"),
    "REQUIRED file containing reference data."
  )(
    "queries_in",
    boost::program_options::value<std::string>(),
    "OPTIONAL file containing query positions."
  )(
    "random_generate_n_attributes",
    boost::program_options::value<int>()->default_value(5),
    "Generate the datasets on the fly of the specified dimension."
  )(
    "random_generate_n_entries",
    boost::program_options::value<int>()->default_value(100000),
    "Generate the datasets on the fly of the specified number of points."
  )(
    "kernel",
    boost::program_options::value<std::string>()->default_value("gaussian"),
    "Kernel function used by KPCA.  One of:\n"
    "  gaussian"
  )(
    "bandwidth",
    boost::program_options::value<double>()->default_value(0.5),
    "OPTIONAL kernel bandwidth, if you set --bandwidth_selection flag, "
    "then the --bandwidth will be ignored."
  )(
    "probability",
    boost::program_options::value<double>()->default_value(0.9),
    "Probability guarantee for the approximation of KPCA."
  )(
    "absolute_error",
    boost::program_options::value<double>()->default_value(1e-6),
    "Absolute error for the approximation of KPCA per each query point."
  )(
    "relative_error",
    boost::program_options::value<double>()->default_value(0.1),
    "Relative error for the approximation of KPCA."
  )(
    "use_memory_mapped_file",
    "Use memory mapped file for out-of-core computations."
  )(
    "memory_mapped_file_size",
    boost::program_options::value<unsigned int>(),
    "The size of the memory mapped file."
  )(
    "prescale",
    boost::program_options::value<std::string>()->default_value("none"),
    "OPTIONAL scaling option. One of:\n"
    "  none, hypercube, standardize"
  );

  boost::program_options::command_line_parser clp(args);
  clp.style(boost::program_options::command_line_style::default_style
            ^ boost::program_options::command_line_style::allow_guessing);
  try {
    boost::program_options::store(clp.options(desc).run(), *vm);
  }
  catch(const boost::program_options::invalid_option_value &e) {
    std::cerr << "Invalid Argument: " << e.what() << "\n";
    exit(0);
  }
  catch(const boost::program_options::invalid_command_line_syntax &e) {
    std::cerr << "Invalid command line syntax: " << e.what() << "\n";
    exit(0);
  }
  catch(const boost::program_options::unknown_option &e) {
    std::cerr << "Unknown option: " << e.what() << "\n";
    exit(0);
  }

  boost::program_options::notify(*vm);
  if(vm->count("help")) {
    std::cout << desc << "\n";
    return true;
  }

  // Validate the arguments. Only immediate dying is allowed here, the
  // parsing is done later.
  if(vm->count("random_generate_n_attributes") > 0) {
    if(vm->count("random_generate_n_entries") == 0) {
      std::cerr << "Missing required --random_generate_n_entries.\n";
      exit(0);
    }
    if((*vm)["random_generate_n_attributes"].as<int>() <= 0) {
      std::cerr << "The --random_generate_n_attributes requires a positive "
                "integer.\n";
      exit(0);
    }
  }
  if(vm->count("random_generate_n_entries") > 0) {
    if(vm->count("random_generate_n_attributes") == 0) {
      std::cerr << "Missing required --random_generate_n_attributes.\n";
      exit(0);
    }
    if((*vm)["random_generate_n_entries"].as<int>() <= 0) {
      std::cerr << "The --random_generate_n_entries requires a positive "
                "integer.\n";
      exit(0);
    }
  }
  if(vm->count("mode") > 0) {
    if((*vm)["mode"].as<std::string>() != "kde" &&
        (*vm)["mode"].as<std::string>() != "kpca") {
      std::cerr << "The mode supports either kde or kpca.\n";
      exit(0);
    }
  }
  if(vm->count("references_in") == 0) {
    std::cerr << "Missing required --references_in.\n";
    exit(0);
  }
  if((*vm)["kernel"].as<std::string>() != "gaussian") {
    std::cerr << "We support only gaussian for the kernel.\n";
    exit(0);
  }
  if(vm->count("bandwidth") > 0 && (*vm)["bandwidth"].as<double>() <= 0) {
    std::cerr << "The --bandwidth requires a positive real number.\n";
    exit(0);
  }
  if(vm->count("bandwidth") == 0) {
    std::cerr << "Missing required --bandwidth.\n";
    exit(0);
  }
  if((*vm)["probability"].as<double>() <= 0 ||
      (*vm)["probability"].as<double>() > 1) {
    std::cerr << "The --probability requires a real number $0 < p <= 1$.\n";
    exit(0);
  }
  if((*vm)["relative_error"].as<double>() < 0) {
    std::cerr << "The --relative_error requires a real number $r >= 0$.\n";
    exit(0);
  }
  if((*vm)["num_kpca_components_in"].as<int>() <= 0) {
    std::cerr << "The --num_kpca_components_in requires an integer > 0.\n";
  }

  // Check whether the memory mapped file is being requested.
  if(vm->count("use_memory_mapped_file") > 0) {

    if(vm->count("memory_mapped_file_size") == 0) {
      std::cerr << "The --used_memory_mapped_file requires an additional "
                "parameter --memory_mapped_file_size.\n";
      exit(0);
    }
    unsigned int memory_mapped_file_size =
      (*vm)["memory_mapped_file_size"].as<unsigned int>();
    if(memory_mapped_file_size <= 0) {
      std::cerr << "The --memory_mapped_file_size needs to be a positive "
                "integer.\n";
      exit(0);
    }

    // Delete the teporary files and put a barrier.
    std::stringstream temporary_file_name;
    temporary_file_name << "tmp_file" << world.rank();
    remove(temporary_file_name.str().c_str());
    world.barrier();

    // Initialize the memory allocator.
    core::table::global_m_file_ = new core::table::MemoryMappedFile();
    core::table::global_m_file_->Init(
      std::string("tmp_file"), world.rank(), world.rank(), 100000000);
  }
  if(vm->count("prescale") > 0) {
    if((*vm)["prescale"].as<std::string>() != "hypercube" &&
        (*vm)["prescale"].as<std::string>() != "standardize" &&
        (*vm)["prescale"].as<std::string>() != "none") {
      std::cerr << "The --prescale needs to be: none or hypercube or " <<
                "standardize.\n";
      exit(0);
    }
  }

  return false;
}

template<typename TableType>
void DistributedKpcaArgumentParser::RandomGenerate(
  boost::mpi::communicator &world, const std::string &file_name,
  int num_dimensions, int num_points, const std::string &prescale_option) {

  // Each process generates its own random data, dumps it to the file,
  // and read its own file back into its own distributed table.
  TableType random_dataset;
  random_dataset.Init(num_dimensions, num_points);
  for(int j = 0; j < num_points; j++) {
    core::table::DensePoint point;
    random_dataset.get(j, &point);
    for(int i = 0; i < num_dimensions; i++) {
      point[i] = core::math::Random(0.1, 1.0);
    }
  }
  printf("Process %d generated %d points in %d dimensionality...\n",
         world.rank(), num_points, num_dimensions);

  // Scale the dataset.
  if(prescale_option == "hypercube") {
    core::table::UnitHypercube::Transform(&random_dataset);
  }
  else if(prescale_option == "standardize") {
    core::table::Standardize::Transform(&random_dataset);
  }
  std::cout << "Scaled the dataset with the option: " <<
            prescale_option << "\n";

  random_dataset.Save(file_name);
}

template<typename DistributedTableType>
bool DistributedKpcaArgumentParser::ParseArguments(
  boost::mpi::communicator &world,
  boost::program_options::variables_map &vm,
  mlpack::distributed_kpca::DistributedKpcaArguments <
  DistributedTableType > *arguments_out) {

  // Parse the reference set and index the tree.
  std::string reference_file_name = vm["references_in"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream reference_file_name_sstr;
    reference_file_name_sstr << vm["references_in"].as<std::string>() <<
                             world.rank();
    reference_file_name = reference_file_name_sstr.str();
    RandomGenerate<typename DistributedTableType::TableType>(
      world, reference_file_name, vm["random_generate_n_attributes"].as<int>(),
      vm["random_generate_n_entries"].as<int>(),
      vm["prescale"].as<std::string>());
  }

  std::cout << "Reading in the reference set: " <<
            reference_file_name << "\n";
  arguments_out->reference_table_ =
    (core::table::global_m_file_) ?
    core::table::global_m_file_->Construct<DistributedTableType>() :
    new DistributedTableType();
  arguments_out->reference_table_->Init(
    reference_file_name, world);

  // Parse the query set and index the tree.
  if(vm.count("queries_in") > 0) {
    std::string query_file_name = vm["queries_in"].as<std::string>();
    if(vm.count("random_generate") > 0) {
      std::stringstream query_file_name_sstr;
      query_file_name_sstr << vm["queries_in"].as<std::string>() <<
                           world.rank();
      query_file_name = query_file_name_sstr.str();
      RandomGenerate<typename DistributedTableType::TableType>(
        world, query_file_name, vm["random_generate_n_attributes"].as<int>(),
        vm["random_generate_n_entries"].as<int>(),
        vm["prescale"].as<std::string>());
    }
    std::cout << "Reading in the query set: " <<
              query_file_name << "\n";
    arguments_out->query_table_ =
      (core::table::global_m_file_) ?
      core::table::global_m_file_->Construct<DistributedTableType>() :
      new DistributedTableType();
    arguments_out->query_table_->Init(query_file_name, world);
    std::cout << "Finished reading in the query set.\n";
    std::cout << "Building the query tree.\n";
  }

  // Parse the bandwidth.
  arguments_out->bandwidth_ = vm["bandwidth"].as<double>();
  if(world.rank() == 0) {
    std::cout << "Bandwidth of " << arguments_out->bandwidth_ << "\n";
  }

  // Parse the relative error and the absolute error.
  arguments_out->absolute_error_ = vm["absolute_error"].as<double>();
  arguments_out->relative_error_ = vm["relative_error"].as<double>();
  if(world.rank() == 0) {
    std::cout << "For each query point $q \\in \\mathcal{Q}$, " <<
              "we will guarantee: " <<
              "$| \\widetilde{G}(q) - G(q) | \\leq "
              << arguments_out->relative_error_ <<
              " \\cdot G(q) + " << arguments_out->absolute_error_ <<
              " | \\mathcal{R} |$ \n";
  }

  // Parse the probability.
  arguments_out->probability_ = vm["probability"].as<double>();
  if(world.rank() == 0) {
    std::cout << "Probability of " << arguments_out->probability_ << "\n";
  }

  // Parse the kernel type.
  arguments_out->kernel_ = vm["kernel"].as< std::string >();
  if(world.rank() == 0) {
    std::cout << "Using the kernel: " << arguments_out->kernel_ << "\n";
  }

  // Parse the KPCA component output file.
  arguments_out->kpca_components_out_ =
    vm["kpca_components_out"].as<std::string>();
  if(vm.count("random_generate_n_entries") > 0) {
    std::stringstream kpca_components_out_sstr;
    kpca_components_out_sstr << vm["kpca_components_out"].as<std::string>() <<
                             world.rank();
    arguments_out->kpca_components_out_ = kpca_components_out_sstr.str();
  }

  // Parse the mode.
  arguments_out->mode_ = vm["mode"].as<std::string>();
  std::cout << "Running in the mode: " << arguments_out->mode_ << ".\n";

  // Parse the number of KPCA components.
  arguments_out->num_kpca_components_in_ =
    (vm["mode"].as<std::string>() == "kde") ?
    1 : vm["num_kpca_components_in"].as<int>();
  if(world.rank() == 0) {
    std::cout << "Requesting " << arguments_out->num_kpca_components_in_ <<
              " kernel PCA components...\n";
  }

  // Parse whether the centering is requested for KPCA or not.
  arguments_out->do_centering_ = (vm.count("do_centering") > 0);
  if(world.rank() == 0 && arguments_out->mode_ == "kpca") {
    if(arguments_out->do_centering_) {
      std::cout << "Doing a centered kernel PCA.\n";
    }
    else {
      std::cout << "Doing a non-centered version of kernel PCA.\n";
    }
  }
  return false;
}

bool DistributedKpcaArgumentParser::ConstructBoostVariableMap(
  boost::mpi::communicator &world,
  int argc,
  char *argv[],
  boost::program_options::variables_map *vm) {

  // Convert C input to C++; skip executable name for Boost.
  std::vector<std::string> args(argv + 1, argv + argc);

  return ConstructBoostVariableMap(world, args, vm);
}
}
}

#endif
