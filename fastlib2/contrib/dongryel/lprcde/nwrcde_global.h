#ifndef NWRCDE_GLOBAL_H
#define NWRCDE_GLOBAL_H

template<typename TKernel, typename ReferenceTree>
class NWRCdeGlobal {

 public:

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module;

  /** @brief The kernel function.
   */
  TKernel kernel;

  /** @brief The relative error desired.
   */
  double relative_error;

  /** @brief The reference dataset.
   */
  Matrix rset;
  
  /** @brief The reference targets.
   */
  Vector rset_targets;
  
  /** @brief The sum of the reference targets.
   */
  double rset_target_sum;
  
  /** @brief The reference tree.
   */
  ReferenceTree *rroot;
  
  /** @brief The permutation mapping indices of references_ to
   *         original order.
   */
  ArrayList<index_t> old_from_new_references;

  OT_DEF_BASIC(NWRCdeGlobal) {
    OT_PTR_NULLABLE(module);
    OT_MY_OBJECT(kernel);
    OT_MY_OBJECT(relative_error);
    OT_MY_OBJECT(rset);
    OT_MY_OBJECT(rset_targets);
    OT_MY_OBJECT(rset_target_sum);
    OT_PTR_NULLABLE(rroot);
    OT_MY_OBJECT(old_from_new_references);
  }

};

#endif
