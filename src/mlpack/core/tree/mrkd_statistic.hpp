/**
 * @file mrkd_statistic.hpp
 *
 * Definition of the policy type for the statistic class.
 *
 * You should define your own statistic that looks like EmptyStatistic.
 */

#ifndef __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP
#define __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP

namespace mlpack {
namespace tree {

/**
 * Statistic for multi-resolution kd-trees.
 */
class MRKDStatistic
{
  public:
    MRKDStatistic()
    :
      dataset(NULL),
      begin(0),
      count(0)
    { }

    ~MRKDStatistic() {}

    /**
     * This constructor is called when a leaf is created.
     *
     * @param dataset Matrix that the tree is being built on.
     * @param begin Starting index corresponding to this leaf.
     * @param count Number of points held in this leaf.
     */
    template<typename MatType>
    MRKDStatistic(const MatType& dataset,
                   const size_t begin,
                   const size_t count)
    :
      dataset(dataset),
      begin(begin),
      count(count)
    {
      centerOfMass = dataset[begin];
      for(int i = begin+1; i < begin+count; ++i)
        centerOfMass += dataset[i];
      centerOfMass /= count;

      sumOfSquaredNorms = 0.0;
      for(int i = begin; i < begin+count; ++i)
        sumOfSquaredNorms += arma::norm(dataset[i], 2);
    }

    /**
     * This constructor is called when a non-leaf node is created.
     * This lets you build fast bottom-up statistics when building trees.
     *
     * @param dataset Matrix that the tree is being built on.
     * @param begin Starting index corresponding to this leaf.
     * @param count Number of points held in this leaf.
     * @param leftStat MRKDStatistic object of the left child node.
     * @param rightStat MRKDStatistic object of the right child node.
     */
    template<typename MatType>
    MRKDStatistic(const MatType& dataset,
                   const size_t begin,
                   const size_t count,
                   const MRKDStatistic& leftStat,
                   const MRKDStatistic& rightStat)
    :
      dataset(dataset),
      begin(begin),
      count(count),
      leftStat(leftStat),
      rightStat(rightStat)
    {
      sumOfSquaredNorms = leftStat.sumOfSquaredNorms + rightStat.sumOfSquaredNorms;

      centerOfMass = ((leftStat.centerOfMass * leftStat.count) +
                      (rightStat.centerOfMass * rightStat.count)) /
                      (leftStat.count + rightStat.count);
    }

    //! The data points this object contains
    const arma::mat* dataset;
    //! The left child 
    const MRKDStatistic* leftStat;
    //! The right child 
    const MRKDStatistic* rightStat;
    //! The initial item in the dataset, so we don't have to make a copy
    const size_t begin;
    //! The number of items in the dataset
    const size_t count;

    // Computed statistics
    //! The center of mass for this dataset
    arma::vec centerOfMass;
    //! The sum of the squared Euclidian norms for this dataset
    double sumOfSquaredNorms;
};

}; // namespace tree
}; // namespace mlpack

#endif // __MLPACK_CORE_TREE_MRKD_STATISTIC_HPP
