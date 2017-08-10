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
 * s.t. \quad v = \sum_{g\in G} v_g
 * \f]
 * This norm is an atom norm, and the dual norm is given by
 * \f[
 * ||y||^*_G := \max_{g\in G} ||y_g||_g^*
 * \f]
 *
 * See Jaggi's paper:
 * @code
 * @inproceedings{Jag:2013Revisiting,
 *  Author = {Jaggi, Martin},
 *  Booktitle = {ICML (1)},
 *  Pages = {427--435},
 *  Title = {Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization.},
 *  Year = {2013}}
 * @endcode
 *
 *  For ConstrStrctGroupSolver to work, we need to use template class GroupType, 
 *  which gives functions:
 *
 *    size_t NumGroups();
 *    double DualNorm(const arma::vec& yk, const int group_ind);
 *    ProjectToGroup(const arma::mat& v, const size_t groupId, arma::vec& y);
 *    void OptimalFromGroup(const arma::mat& v, const size_t groupId, arma::mat& s);
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
   * @param groupExtractor Class used to project to a group, recovery from a
   *                       group, and compute norm in each group.
   */
  ConstrStructGroupSolver(GroupType& groupExtractor) :
    groupExtractor(groupExtractor)
  { /* Nothing to do */ }

  /**
   * Optimizer of structure group ball constrained Problem for FrankWolfe.
   *
   * @param v Input local gradient.
   * @param s Output optimal solution in the constrained atom domain.
   */
  void Optimize(const arma::mat& v, arma::mat& s)
  {
    size_t nGroups = groupExtractor.NumGroups();
    double dualNorm = 0;
    size_t optimalGroup = 1;

    // Find the optimal group.
    for (size_t i = 1; i <= nGroups; ++i)
    {
      arma::vec y;
      groupExtractor.ProjectToGroup(v, i, y);
      double newNorm = groupExtractor.DualNorm(y, i);

      // Find the group with largest dual norm.
      if (newNorm > dualNorm)
      {
        optimalGroup = i;
        dualNorm = newNorm;
      }
    }

    groupExtractor.OptimalFromGroup(v, optimalGroup, s);
  }

 private:
  //! Information and methods for groups.
  GroupType& groupExtractor;
};

/**
 * Implementation of Structured Group. The projection to each group is using
 * restriction of vector support here, and the norm in each group is using lp
 * norm.
 */
class GroupLpBall
{
 public:
  /**
   * Construct the lp ball group extractor class.
   *
   * @param p lp ball.
   * @param dimOrig dimension of the original vector.
   * @param groupIndicesList vector of support indices lists of each group.
   */
  GroupLpBall(const double p,
              const size_t dimOrig,
              std::vector<arma::uvec> groupIndicesList):
    p(p), numGroups(groupIndicesList.size()),
    dimOrig(dimOrig),
    groupIndicesList(groupIndicesList),
    lpBallSolver(p)
  {/* Nothing to do. */}

  /**
   * Projection to specific group.
   *
   * @param v input vector to be projected.
   * @param groupId input ID number of the group, start from 1.
   * @param y output projection of the vector to specific group.
   */
  void ProjectToGroup(const arma::mat& v, const size_t groupId, arma::vec& y)
  {
    arma::uvec& indList = groupIndicesList[groupId - 1];
    size_t dim = indList.n_elem;
    y.set_size(dim);

    for (size_t i = 0; i < dim; ++i)
      y(i) = v(indList(i));
  }

  /**
   * Get optimal atom, which belongs to specific group.
   * See Jaggi's paper for details.
   *
   * @param v input gradient vector.
   * @param groupId optimal atom belongs to this group.
   * @param s output optimal atom.
   */
  void OptimalFromGroup(const arma::mat& v, const size_t groupId, arma::mat& s)
  {
    // Project v to group.
    arma::vec yk;
    ProjectToGroup(v, groupId, yk);

    // Optimize in this group.
    arma::vec sProj(yk.n_elem);
    lpBallSolver.Optimize(yk, sProj);

    // Recover s to the original dimension.
    arma::uvec& indList = groupIndicesList[groupId - 1];
    size_t dim = indList.n_elem;  // dimension of the group.
    s.zeros(dimOrig, 1);

    for (size_t i = 0; i < dim; ++i)
      s(indList(i)) = sProj(i);
  }

  //! Get the number of groups.
  size_t NumGroups() const {return numGroups;}
  //! Modify the number of groups.
  size_t& NumGroups() {return numGroups;}

  /**
   * Compute the q-norm of yk, 1/p+1/q=1.
   *
   * @param yk compute the q-norm of yk.
   * @param groupId group ID number.
   */
  double DualNorm(const arma::vec& yk, const int groupId)
  {
    if (p == std::numeric_limits<double>::infinity())
    {
      // inf-norm, return 1-norm
      return arma::norm(yk, 1);
    }
    else if (p > 1.0)
    {
      // p norm, return q-norm
      double q = 1.0 / (1.0 - 1.0/p);
      return  arma::norm(yk, q);
    }
    else if (p == 1.0)
    {
      // 1-norm, return inf-norm
      return arma::norm(yk, "inf");
    }
    else
    {
      Log::Fatal << "Wrong norm p!" << std::endl;
      return 0.0;
    }
  }

 private:
  //! lp norm, 1<=p<=inf;
  //! use std::numeric_limits<double>::infinity() for inf norm.
  double p;

  //! Number of groups.
  size_t numGroups;

  //! Original Problem Dimension.
  size_t dimOrig;

  //! Indices list of each group, indices start from 0.
  std::vector<arma::uvec> groupIndicesList;

  //! Each group uses lp norm
  ConstrLpBallSolver lpBallSolver;
};


} // namespace optimization
} // namespace mlpack



#endif
