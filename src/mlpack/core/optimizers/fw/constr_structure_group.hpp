/**
 * @file constr_structure_group.hpp
 * @author Chenzhe Diao
 *
 * Solve the linear constrained problem, where the constrained domain are atom
 * domains defined by unit balls under structured group norm.
 * Used as LinearConstrSolverType in FrankWolfe.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_CONSTR_STRUCTURE_GROUP_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_CONSTR_STRUCTURE_GROUP_HPP

#include <mlpack/prereqs.hpp>
#include "constr_lpball.hpp"

namespace mlpack {
namespace optimization {

/**
 * Linear Constrained Solver for FrankWolfe. Constrained domain given in the
 * form of unit ball of different structured group. That is, given original
 * vector \f$ v \f$ in high dimensional space, suppose we can map it into
 * different smaller dimensional spaces (decomposing the information):
 * \f[
 * v \rightarrow v_g, \qquad g\in G.
 * \f]
 *
 * For example, each group corresponds to a specific set of support subsets,
 * as in GroupLpBall class. Also, a norm would be equipped for each group:
 * \f$ || v_g ||_g \f$, for example lp norm could be used, as in GroupLpBall
 * class. Now, the norm defined for the original vector is:
 * \f[
 * ||v||_G := \min_{v_g} \sum_{g\in G} ||v_g||_g, \qquad
 * s.t. v = \sum_{g\in G} v_g
 * \f]
 * This norm is an atom norm, and the dual norm is given by
 * \f[
 * ||y||^*_G := \max_{g\in G} ||y_g||_g^*
 * \f]
 *
 *
 *  For ConstrStrctGroupSolver to work, we need to use template class GroupType, 
 *  which gives functions:
 *
 *      int NumGroups();
 *      double DualNorm(const arma::vec& yk, const int group_ind)
 *	    arma::vec OptimalFromGroup(const arma::vec& yk, const size_t GroupId)
 *      arma::vec ProjectToGroup(const arma::vec& v, const size_t GroupId);
 *
 *
 * @tparam GroupType Class that implements functions to map original vectors to
 *                   each group, and to solve linear optimization problem in the
 *                   unit ball defined by the norm of each group.
 */
template<typename GroupType>
class ConstrStructGroupSolver
{
 public:
  /**
   * Construct the structure group optimization solver.
   *
   * @param group_extractor Class used to project to a group, recovery from a
   * group, and compute norm in each group.
   */
  ConstrStructGroupSolver(GroupType& group_extractor) :
    group_extractor(group_extractor)
  { /* Nothing to do */ }

  /**
   * Optimizer of structure group ball constrained Problem for FrankWolfe.
   *
   * @param v Input local gradient.
   * @param s Output optimal solution in the constrained domain.
   */
  void Optimize(const arma::mat& v, arma::mat& s)
  {
    int nGroups = group_extractor.NumGroups();
    double dualnorm = 0;
    int optimal_group = 1;

    // Find the optimal group
    for (size_t i=1; i<= nGroups; ++i)
    {
      arma::mat y = group_extractor.ProjectToGroup(v, i);
      double newnorm = group_extractor.DualNorm(y, i);

      // Find the group with largest dual norm.
      if (newnorm > dualnorm)
      {
        optimal_group = i;
        dualnorm = newnorm;
      }
    }

    arma::mat y_opt = group_extractor.ProjectToGroup(optimal_group);
    s = group_extractor.OptimalFromGroup(y_opt, optimal_group);

  }

 private:
  GroupType& group_extractor;
};

/**
 * Implementation of Structured Group. The projection to each group is using
 * restriction of vector support here, and the norm in each group is using lp
 * norm.
 *
 */
class GroupLpBall
{
 public:
  GroupLpBall(const double p,
              const int dimOrig,
              std::vector<arma::uvec> groupIndList):
    p(p), numGroups(groupIndList.size()),
    dimOrig(dimOrig),
    groupIndList(groupIndList),
    lpBallGroup(p)
  {/* Nothing to do. */}

  arma::vec ProjectToGroup(const arma::vec& v, const size_t groupId)
  {
    arma::uvec& indList = groupIndList[groupId-1];
    size_t dim = indList.n_elem;
    arma::vec y(dim);

    for (size_t i = 0; i < dim; ++i)
    {
      y(i) = v(indList(i));
    }

    return y;
  }

  arma::vec OptimalFromGroup(const arma::vec& yk, const size_t groupId)
  {
    arma::vec sProj;
    lpBallGroup.Optimize(yk, sProj);

    arma::uvec& indList = groupIndList[groupId-1];
    size_t dim = indList.n_elem;
    arma::vec s = arma::zeros<arma::vec>(dimOrig);

    for (size_t i = 0; i < dim; ++i)
    {
      s(indList(i)) = sProj(i);
    }
    return s;
  }
    
  int NumGroups() const {return numGroups;}
  int& NumGroups() {return numGroups;}
    
  // group_ind start from 1
  double DualNorm(const arma::vec& yk, const int groupInd)
  {
    if (p == std::numeric_limits<double>::infinity()) {
      // inf-norm, return 1-norm
      return sum(abs(yk));
    }
    else if (p > 1.0) {
      // p norm, return q-norm
      double q = 1 / (1 - 1/p);
      return  std::pow(sum(pow(abs(yk), q)), 1/q);
    }
    else if (p == 1.0) {
      return max(abs(yk));
    }
    else {
      Log::Fatal << "Wrong norm p!" << std::endl;
      return 0.0;
    }
  }

 private:
  //! lp norm, 1<=p<=inf;
  //! use std::numeric_limits<double>::infinity() for inf norm.
  double p;

  //! Number of groups.
  int numGroups;

  //! Original Problem Dimension.
  int dimOrig;

  //! Indices list of each group, indices start from 0.
  std::vector<arma::uvec> groupIndList;

  //! Each group uses a lp norm
  ConstrLpBallSolver lpBallGroup;
};


}// namespace optimization
}// namespace mlpack



#endif
