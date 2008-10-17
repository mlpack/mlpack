#ifndef NWRCDE_GLOBAL_H
#define NWRCDE_GLOBAL_H

template<typename TKernelAux, typename ReferenceTree>
class NWRCdeGlobal {

 public:

  /** @brief The pointer to the module holding the parameters.
   */
  struct datanode *module;

  /** @brief The kernel function.
   */
  TKernelAux kernel_aux;

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

  // It is important not to include the module pointer because it will
  // be freed by fx_done()!
  OT_DEF_BASIC(NWRCdeGlobal) {
    OT_MY_OBJECT(kernel_aux);
    OT_MY_OBJECT(relative_error);
    OT_MY_OBJECT(rset);
    OT_MY_OBJECT(rset_targets);
    OT_MY_OBJECT(rset_target_sum);
    OT_PTR_NULLABLE(rroot);
    OT_MY_OBJECT(old_from_new_references);
  }

};

#endif
