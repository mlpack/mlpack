#ifndef INSIDE_TOOLS_H
#error "This is not a public header file!"
#endif


void RemoveRows(const mat& X, uvec rows_to_remove, mat& X_mod) {

  u32 n_cols = X.n_cols;
  u32 n_rows = X.n_rows;
  u32 n_to_remove = rows_to_remove.n_elem;
  u32 n_to_keep = n_rows - n_to_remove;
  
  if(n_to_remove == 0) {
    X_mod = X;
  }
  else {
    X_mod.set_size(n_to_keep, n_cols);

    u32 cur_row = 0;
    u32 remove_ind = 0;
    // first, check 0 to first row to remove
    if(rows_to_remove(0) > 0) {
      // note that this implies that n_rows > 1
      u32 height = rows_to_remove(0);
      X_mod(span(cur_row, cur_row + height - 1), span::all) =
	X(span(0, rows_to_remove(0) - 1), span::all);
      cur_row += height;
    }
    // now, check i'th row to remove to (i + 1)'th row to remove, until i = penultimate row
    while(remove_ind < n_to_remove - 1) {
      u32 height = 
	rows_to_remove[remove_ind + 1]
	- rows_to_remove[remove_ind]
	- 1;
      if(height > 0) {
	X_mod(span(cur_row, cur_row + height - 1), 
	      span::all) =
	  X(span(rows_to_remove[remove_ind] + 1,
		 rows_to_remove[remove_ind + 1] - 1), 
	    span::all);
	cur_row += height;
      }
      remove_ind++;
    }
    // now that i is last row to remove, check last row to remove to last row
    if(rows_to_remove[remove_ind] < n_rows - 1) {
      X_mod(span(cur_row, n_to_keep - 1), 
	    span::all) = 
	X(span(rows_to_remove[remove_ind] + 1, n_rows - 1), 
	  span::all);
    }
  }
}
