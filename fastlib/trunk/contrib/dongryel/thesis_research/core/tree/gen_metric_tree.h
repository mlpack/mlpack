/** @file gen_metric.h
 *
 *  The generic metric tree builder.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CORE_TREE_GEN_METRIC_TREE_H
#define CORE_TREE_GEN_METRIC_TREE_H

#include <armadillo>
#include <vector>
#include <boost/mpi.hpp>
#include <boost/mpi/operations.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/tuple/tuple.hpp>
#include <omp.h>

#include "core/math/math_lib.h"
#include "core/parallel/distributed_tree_util.h"
#include "core/tree/ball_bound.h"
#include "core/tree/general_spacetree.h"
#include "core/table/dense_matrix.h"
#include "core/table/memory_mapped_file.h"

namespace boost {
namespace serialization {

template <class Archive, typename MetricType>
void serialize(
  Archive & ar,
  boost::tuple< MetricType, core::tree::BallBound, int > &c,
  const unsigned int version) {
  ar & c.get<0>();
  ar & c.get<1>();
  ar & c.get<2>();
}

template <class Archive, typename T>
void serialize(
  Archive & ar, std::pair<arma::vec, T> &c, const unsigned int version) {
  ar & c.first;
  ar & c.second;
}
}
}

namespace core {
namespace parallel {

/** @brief A class that combines two bounding balls and produces the
 *         tightest ball that contains both.
 */
template<typename MetricType>
class BallBoundCombine:
  public std::binary_function <
  boost::tuple< MetricType, core::tree::BallBound, int >,
  boost::tuple< MetricType, core::tree::BallBound, int >,
    boost::tuple< MetricType, core::tree::BallBound, int > > {
  public:
    const boost::tuple< MetricType, core::tree::BallBound, int > operator()(
      const boost::tuple< MetricType, core::tree::BallBound, int > &a,
      const boost::tuple< MetricType, core::tree::BallBound, int > &b) const {

      boost::tuple< MetricType, core::tree::BallBound, int > combined_ball;
      combined_ball.template get<0>() = a.template get<0>();
      combined_ball.template get<1>().Init(a.template get<1>().dim());

      // Compute the weighted sum of the two pivots
      arma::vec &bound_ref = combined_ball.template get<1>().center();
      const arma::vec &left_bound_ref = a.template get<1>().center();
      const arma::vec &right_bound_ref = b.template get<1>().center();
      bound_ref = a.template get<2>() * left_bound_ref +
                  b.template get<2>() * right_bound_ref;
      combined_ball.template get<2>() =
        a.template get<2>() + b.template get<2>();
      bound_ref =
        (1.0 / static_cast<double>(combined_ball.template get<2>())) *
        bound_ref;

      return combined_ball;
    }
};

class ChooseMaxPoint:
  public std::binary_function <
  std::pair< arma::vec, double> , std::pair< arma::vec, double>,
    std::pair< arma::vec, double>  > {

  public:
    const std::pair< arma::vec, double> operator()(
      const std::pair< arma::vec, double> &first_point_pair,
      const std::pair< arma::vec, double> &second_point_pair) const {
      std::pair< arma::vec, double > random_point_pair;
      if(first_point_pair.second > second_point_pair.second) {
        random_point_pair = first_point_pair;
      }
      else {
        random_point_pair = second_point_pair;
      }
      return random_point_pair;
    }
};

/** @brief A class that does tie breaking randomly and returns one
 *         of the two points.
 */
class ChooseRandomPoint:
  public std::binary_function <
  std::pair<arma::vec, int>, std::pair<arma::vec, int>,
    std::pair<arma::vec, int> > {

  public:
    const std::pair<arma::vec, int> operator()(
      const std::pair< arma::vec, int>  &first_point,
      const std::pair< arma::vec, int>  &second_point) const {
      std::pair<arma::vec, int> random_point;
      random_point.second = first_point.second + second_point.second;
      double first_point_threshold =
        static_cast<double>(first_point.second) /
        static_cast<double>(random_point.second);
      if(core::math::Random<double>() <= first_point_threshold) {
        random_point.first = first_point.first;
      }
      else {
        random_point.first = second_point.first;
      }
      return random_point;
    }
};
}
}

namespace boost {
namespace mpi {

/** @brief BallBoundCombine function is a commutative reduction
 *         operator.
 */
template<>
template<typename MetricType>
class is_commutative <
  core::parallel::BallBoundCombine<MetricType>,
  boost::tuple< MetricType, core::tree::BallBound, int >  > :
  public boost::mpl::true_ {

};

/** @brief RandomPointChoose function is a commutative reduction
 *         operator.
 */
template<>
class is_commutative <
  core::parallel::ChooseRandomPoint, std::pair< arma::vec, int > > :
  public boost::mpl::true_ {

};

template<>
class is_commutative <
  core::parallel::ChooseMaxPoint, std::pair< arma::vec, double > > :
  public boost::mpl::true_ {

};
}
}

namespace core {
namespace tree {

/** @brief The general metric tree specification.
 */
template<typename IncomingStatisticType>
class GenMetricTree {
  public:

    typedef core::tree::BallBound BoundType;

    typedef IncomingStatisticType StatisticType;

  private:

    /** @brief Computes the furthest point from the given pivot and
     *         finds out the index.
     */
    template<typename MetricType>
    static int FurthestColumnIndex_(
      const MetricType &metric_in,
      const arma::vec &pivot,
      const arma::mat &matrix,
      int begin, int count,
      double *furthest_distance) {

      int furthest_index = -1;
      int end = begin + count;
      *furthest_distance = -1.0;

      int num_threads =
        std::max(1, omp_get_max_threads() / omp_get_num_threads());
      omp_set_num_threads(num_threads);
      #pragma omp parallel
      {
        // Local variables used for reduction.
        int local_furthest_index = -1;
        double local_furthest_distance = -1.0;

        #pragma omp for nowait
        for(int i = begin; i < end; i++) {
          arma::vec point;
          core::table::MakeColumnVector(matrix, i, &point);
          double distance_between_center_and_point =
            metric_in.Distance(pivot, point);

          if(local_furthest_distance < distance_between_center_and_point) {
            local_furthest_distance = distance_between_center_and_point;
            local_furthest_index = i;
          }
        } // end of for-loop.

        // Reduction.
        #pragma omp critical
        {
          if(local_furthest_distance > (*furthest_distance)) {
            *furthest_distance = local_furthest_distance;
            furthest_index = local_furthest_index;
          }
        } // end of omp critical.
      } // end of omp parallel.
      return furthest_index;
    }

  public:

    template<typename MetricType>
    static void FindBoundFromMatrix(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int first, int count, BoundType *bounds) {

      MakeLeafNode(metric_in, matrix, first, count, bounds);
    }

    /** @brief The parallel MPI version of finding the bound for which
     *         the reduction is done over a MPI communicator.
     */
    template<typename MetricType>
    static void FindBoundFromMatrix(
      boost::mpi::communicator &comm,
      const MetricType &metric_in,
      const arma::mat &matrix,
      BoundType *combined_bound) {

      // Each MPI process finds a local bound.
      boost::tuple<MetricType, BoundType, int> local_bound;
      local_bound.template get<0>() = metric_in;
      local_bound.template get<2>() = matrix.n_cols;
      local_bound.template get<1>().Init(matrix.n_rows);
      FindBoundFromMatrix(
        metric_in, matrix, 0, matrix.n_cols,
        & (local_bound.template get<1>()));

      // The global bound.
      boost::tuple<MetricType, BoundType, int> global_bound;

      // Call reduction.
      boost::mpi::all_reduce(
        comm, local_bound, global_bound,
        core::parallel::BallBoundCombine<MetricType>());

      // Copy out the final bound.
      combined_bound->Copy(global_bound.template get<1>());
    }

    /** @brief Makes a leaf node in the metric tree.
     */
    template<typename MetricType>
    static void MakeLeafNode(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int begin, int count, BoundType *bounds) {

      // Clear the bound to zero.
      bounds->center().zeros();

      int end = begin + count;
      arma::vec &bound_ref = bounds->center();

      // The number of threads.
      int num_threads =
        std::max(1, omp_get_max_threads() / omp_get_num_threads());
      omp_set_num_threads(num_threads);
      #pragma omp parallel
      {
        arma::vec local_sum;
        local_sum.zeros(bound_ref.n_elem);

        #pragma omp for nowait
        for(int i = begin; i < end; i++) {
          arma::vec col_point;
          core::table::MakeColumnVector(matrix, i, &col_point);
          local_sum += col_point;
        }

        // Final reduction.
        #pragma omp critical
        {
          bound_ref += local_sum;
        } // end omp critical.
      }

      // Divide by the number of points.
      bound_ref = (1.0 / static_cast<double>(count)) * bound_ref;

      double furthest_distance;
      FurthestColumnIndex_(
        metric_in, bounds->center(), matrix, begin, count, &furthest_distance);
      bounds->set_radius(furthest_distance);
    }

    template<typename MetricType, typename TreeType>
    static void CombineBounds(
      const MetricType &metric_in,
      arma::mat &matrix,
      TreeType *node, TreeType *left, TreeType *right) {

      // Compute the weighted sum of the two pivots
      arma::vec &bound_ref = node->bound().center();
      arma::vec &left_bound_ref = left->bound().center();
      arma::vec &right_bound_ref = right->bound().center();
      bound_ref = left->count() * left_bound_ref +
                  right->count() * right_bound_ref;
      bound_ref =
        (1.0 / static_cast<double>(node->count())) * bound_ref;

      double left_max_dist, right_max_dist;
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, left->begin(),
        left->count(), &left_max_dist);
      FurthestColumnIndex_(
        metric_in, node->bound().center(), matrix, right->begin(),
        right->count(), &right_max_dist);
      node->bound().set_radius(std::max(left_max_dist, right_max_dist));
    }

    template<typename MetricType>
    static void ComputeMemberships(
      const MetricType &metric_in,
      const arma::mat &matrix,
      int first, int end,
      BoundType &left_bound, BoundType &right_bound,
      int *left_count, std::deque<bool> *left_membership,
      bool favor_balance = false ) {

      left_membership->resize(end - first);
      *left_count = 0;

      #pragma omp parallel
      {
        int local_left_count = 0;

        #pragma omp for
        for(int left = first; left < end; left++) {

          // Make alias of the current point.
          arma::vec point;
          core::table::MakeColumnVector(matrix, left, &point);

          // Compute the distances from the two pivots.
          double distance_from_left_pivot =
            metric_in.Distance(point, left_bound.center());
          double distance_from_right_pivot =
            metric_in.Distance(point, right_bound.center());

          // We swap if the point is further away from the left pivot.
          if(distance_from_left_pivot > distance_from_right_pivot) {
            (*left_membership)[left - first] = false;
          }
          else {
            (*left_membership)[left - first] = true;
            local_left_count++;
          }
        }

        // Final reduction.
        #pragma omp critical
        {
          (*left_count) += local_left_count;
        } // end of omp critical.
      } // end of omp parallel.

      if( favor_balance ) {
	int count = end - first;
	if( (*left_count) <= count / 4 ) {
	  
	  // Steal from the right.
	  for(int i = 0; i < count; i++) {
	    if( ! ( (*left_membership)[i] )  ) {
	      (*left_count)++;
	      (*left_membership)[i] = true;
	    }
	    int current_right_count = count - (*left_count);
	    if( abs( (*left_count ) - current_right_count ) < count / 8 ) {
	      break;
	    }
	  }
	}
	else if( (*left_count) >= count / 4 * 3 ) {
	  
	  // Steal from the left.
	  for(int i = 0; i < count; i++) {
	    if( (*left_membership)[i] ) {
	      (*left_count)--;
	      (*left_membership)[i] = false;
	    }
	    int current_right_count = count - (*left_count);
	    if( abs( (*left_count ) - current_right_count ) < count / 8 ) {
	      break;
	    }
	  }
	}
      }
    }

    template<typename MetricType>
    static bool AttemptSplitting(
      boost::mpi::communicator &comm,
      const MetricType &metric_in,
      const BoundType &bound,
      const arma::mat &matrix_in,
      int *left_count,
      std::deque<bool> *left_membership) {

      // Pick a random point across all processes.
      int local_random_row =
        core::math::RandInt(0, static_cast<int>(matrix_in.n_cols));
      std::pair< arma::vec, int > local_random_row_vec;
      std::pair< arma::vec, int > global_random_row_vec;
      core::table::MakeColumnVector(
        matrix_in, local_random_row, &local_random_row_vec.first);
      local_random_row_vec.second = matrix_in.n_cols;

      // Call all reduction.
      boost::mpi::all_reduce(
        comm, local_random_row_vec, global_random_row_vec,
        core::parallel::ChooseRandomPoint());

      // Given the randomly chosen vectors, find the furthest point.
      double local_furthest_distance;
      int local_furthest_from_random_row =
        FurthestColumnIndex_(
          metric_in, global_random_row_vec.first, matrix_in,
          0, matrix_in.n_cols, &local_furthest_distance);
      std::pair< arma::vec, double > local_furthest_from_random_row_vec;
      std::pair< arma::vec, double > global_furthest_from_random_row_vec;
      core::table::MakeColumnVector(
        matrix_in,
        local_furthest_from_random_row,
        &local_furthest_from_random_row_vec.first);
      local_furthest_from_random_row_vec.second = local_furthest_distance;
      boost::mpi::all_reduce(
        comm, local_furthest_from_random_row_vec,
        global_furthest_from_random_row_vec,
        core::parallel::ChooseMaxPoint());

      // Given the furthest point, find its furthest point.
      double local_furthest_from_furthest_distance;
      int local_furthest_from_furthest_random_row =
        FurthestColumnIndex_(
          metric_in, global_furthest_from_random_row_vec.first, matrix_in,
          0, matrix_in.n_cols, &local_furthest_from_furthest_distance);
      std::pair< arma::vec, double >
      local_furthest_from_furthest_random_row_vec;
      std::pair< arma::vec, double >
      global_furthest_from_furthest_random_row_vec;
      core::table::MakeColumnVector(
        matrix_in,
        local_furthest_from_furthest_random_row,
        &local_furthest_from_furthest_random_row_vec.first);
      local_furthest_from_furthest_random_row_vec.second =
        local_furthest_from_furthest_distance;
      boost::mpi::all_reduce(
        comm, local_furthest_from_furthest_random_row_vec,
        global_furthest_from_furthest_random_row_vec,
        core::parallel::ChooseMaxPoint());

      if(global_furthest_from_furthest_random_row_vec.second <
          std::numeric_limits<double>::epsilon()) {
        return false;
      }

      // Assign the point on the local process using the splitting
      // value.
      BoundType left_bound, right_bound;
      left_bound.center() = global_furthest_from_random_row_vec.first;
      right_bound.center() = global_furthest_from_furthest_random_row_vec.first;
      ComputeMemberships(
        metric_in, matrix_in, 0, matrix_in.n_cols, left_bound, right_bound,
        left_count, left_membership, true );

      return true;
    }

    template<typename MetricType, typename TreeType, typename IndexType>
    static bool AttemptSplitting(
      const MetricType &metric_in,
      arma::mat &matrix,
      arma::mat &weights,
      TreeType *node, TreeType **left,
      TreeType **right, int leaf_size, IndexType *old_from_new,
      core::table::MemoryMappedFile *m_file_in) {

      // Pick a random row.
      int random_row = core::math::RandInt(
                         node->begin(), node->begin() + node->count());
      arma::vec random_row_vec;
      core::table::MakeColumnVector(matrix, random_row, & random_row_vec);

      // Now figure out the furthest point from the random row picked
      // above.
      double furthest_distance;
      int furthest_from_random_row =
        FurthestColumnIndex_(
          metric_in, random_row_vec, matrix, node->begin(), node->count(),
          &furthest_distance);
      arma::vec furthest_from_random_row_vec;
      core::table::MakeColumnVector(
        matrix, furthest_from_random_row, &furthest_from_random_row_vec);

      // Then figure out the furthest point from the furthest point.
      double furthest_from_furthest_distance;
      int furthest_from_furthest_random_row =
        FurthestColumnIndex_(
          metric_in, furthest_from_random_row_vec, matrix, node->begin(),
          node->count(), &furthest_from_furthest_distance);
      arma::vec furthest_from_furthest_random_row_vec;
      core::table::MakeColumnVector(
        matrix, furthest_from_furthest_random_row,
        &furthest_from_furthest_random_row_vec);

      // Allocate the left and the right.
      *left = (m_file_in) ?
              m_file_in->Construct<TreeType>() : new TreeType();
      *right = (m_file_in) ?
               m_file_in->Construct<TreeType>() : new TreeType();
      (*left)->bound().center() = furthest_from_random_row_vec;
      (*right)->bound().center() = furthest_from_furthest_random_row_vec;
      int left_count = 0;
      if(furthest_from_furthest_distance <
          std::numeric_limits<double>::epsilon()) {
        return false;
      }
      else {
        left_count = TreeType::MatrixPartition(
                       metric_in, matrix, weights, node->begin(), node->count(),
                       (*left)->bound(), (*right)->bound(), old_from_new);
      }
      (*left)->Init(node->begin(), left_count);
      (*right)->Init(node->begin() + left_count, node->count() - left_count);
      return true;
    }
};
}
}

#endif
