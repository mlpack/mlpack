/*
 * =====================================================================================
 * 
 *       Filename:  nmf_tree.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  08/14/2008 05:40:10 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef NMF_TREE_H_
#define NMF_TREE_H_

typedef BinarySpaceTree<DHrectBound<2>, Matrix> TreeType;

template<typename TKdTree, typename T>
class NmfTreeConstructor {
 public:

  void Init(fx_module *module, 
            ArrayList<index_t> &rows,
            ArrayList<index_t> &columns,
            ArrayList<double> &values); 
  void MakeNmfTree();   
  void MakeNmfTreeMidpointSelective(TKdTree *node);  
  void SelectSplitKdTreeMidpoint(TKdTree *node, 
      GenVector<T>& split_dimensions);

 private:
  fx_module *module_;
  RelaxedNmf opt_fun_;
  LBfgs<RelaxedNmf> l_bfgs_engine_;
  GenMatrix<T> data_matrix_;
  GenMatrix<T> w_matrix_;
  GenMatrix<T> h_matrix_;
  GenMatrix<T> lower_bound_;
  GenMatrix<T> upper_bound_;
  index_t leaf_size_;
  index_t w_offset_;
  index_t h_offset_;
  GenVector<index_t> old_from_new_w_; 
  GenVector<index_t> old_from_new_h_; 
  GenVector<index_t> new_from_old_;
  ArrayList<index_t> &rows_;
  ArrayList<index_t> &columns_;
  ArrayList<double> &values_;  
  TKdTree *parent_w_;
  TKdTree *parent_h_;
};

#include "nmf_tree_impl.h"

#endif  // NMF_TREE_H_
