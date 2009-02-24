#ifndef KACZMARZ_METHOD_H
#define KACZMARZ_METHOD_H

class KaczmarzMethod {

 public:
  
  /** @brief Solves the A^T x = b.
   */
  static void SolveTransInit(const Matrix &linear_system, 
			     const Vector &right_hand_side, Vector *solution) {

    // The vector solution to the column-oriented system.
    solution->Init(linear_system.n_cols());
    solution->SetZero();

    // The current estimate of the residual.
    Vector residual;
    residual.Copy(right_hand_side);

    // The termination flag.
    bool done_flag = false;
    
    // The current iteration number.
    index_t iteration_number = 0;
    index_t current_index = 0;

    do {

      // The current column in consideration and its squared L2 norm.
      const double *current_column = linear_system.GetColumnPtr(current_index);
      double squared_norm_current_column = 
	la::Dot(linear_system.n_rows(), current_column, current_column);
      
      // The dot product between the current column and the current
      // solution.
      double proj_solution_to_current_column = 
	la::Dot(linear_system.n_rows(), current_column, solution->ptr());

      // Scalar factor to be multiplied for the current column for
      // updating the solution.
      double factor = (right_hand_side[current_index] - 
		       proj_solution_to_current_column) / 
	squared_norm_current_column;

      // Update the solution.
      la::AddExpert(linear_system.n_cols(), factor, current_column,
		    solution->ptr());

      // Increment the iteration number and update the index to be
      // considered in the next iteration.
      iteration_number++;
      current_index++;
      if(current_index == linear_system.n_cols()) {
	current_index = 0;
      }

      // For now, let's terminate when the iteration number reaches
      // the number of columns...
      if(iteration_number == linear_system.n_cols() * 4) {
	done_flag = true;
      }

    } while(!done_flag);
  }

  static void RandomizedSolveTransInit(const Matrix &linear_system,
				       const Vector &right_hand_side,
				       Vector *solution) {

  }
};

#endif
