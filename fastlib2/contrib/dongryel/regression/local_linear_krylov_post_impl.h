// Make sure this file is included only in local_linear_krylov.h. This
// is not a public header file!
#ifndef INSIDE_LOCAL_LINEAR_KRYLOV_H
#error "This file is not a public header file!"
#endif

template<typename TKernel>
void LocalLinearKrylov<TKernel>::FinalizeQueryTreeRightHandSides_
(Tree *qnode) {

  LocalLinearKrylovStat &q_stat = qnode->stat();

  if(qnode->is_leaf()) {
    for(index_t q = qnode->begin(); q < qnode->end(); q++) {
      
      // Get the column vectors accumulating the sums to update.
      double *q_right_hand_sides_l = right_hand_sides_l_.GetColumnPtr(q);
      double *q_right_hand_sides_e = right_hand_sides_e_.GetColumnPtr(q);
      double *q_right_hand_sides_u = right_hand_sides_u_.GetColumnPtr(q);
      

      // Incorporate the postponed information.
      la::AddTo(row_length_,
		(q_stat.postponed_right_hand_sides_l_).ptr(),
		q_right_hand_sides_l);
      la::AddTo(row_length_,
		(q_stat.postponed_right_hand_sides_e_).ptr(),
		q_right_hand_sides_e);
      la::AddTo(row_length_,
		(q_stat.postponed_right_hand_sides_u_).ptr(),
		q_right_hand_sides_u);

      // Maybe I should normalize the sums here to prevent overflow...
    }
  }
  else {
    
    LocalLinearKrylovStat &q_left_stat = qnode->left()->stat();
    LocalLinearKrylovStat &q_right_stat = qnode->right()->stat();

    // Push down approximations
    la::AddTo(q_stat.postponed_right_hand_sides_l_,
	      &(q_left_stat.postponed_right_hand_sides_l_));
    la::AddTo(q_stat.postponed_right_hand_sides_l_,
	      &(q_right_stat.postponed_right_hand_sides_l_));
    la::AddTo(q_stat.postponed_right_hand_sides_e_,
              &(q_left_stat.postponed_right_hand_sides_e_));
    la::AddTo(q_stat.postponed_right_hand_sides_e_,
              &(q_right_stat.postponed_right_hand_sides_e_));
    la::AddTo(q_stat.postponed_right_hand_sides_u_,
              &(q_left_stat.postponed_right_hand_sides_u_));
    la::AddTo(q_stat.postponed_right_hand_sides_u_,
              &(q_right_stat.postponed_right_hand_sides_u_));

    FinalizeQueryTreeRightHandSides_(qnode->left());
    FinalizeQueryTreeRightHandSides_(qnode->right());
  }
}
