#ifndef HMM_KERNEL_UTILS_H
#define HMM_KERNEL_UTILS_H


void ComputePqq(const ArrayList<Matrix> &p_qq_t,
		Matrix *p_p_qq) {
  Matrix &p_qq = *p_p_qq;

  int n_states = p_qq_t.size();
  int sequence_length_minus_1 = p_qq_t[0].n_cols() - 1;

  p_qq.Init(n_states, n_states);
  p_qq.SetZero();
  for(int i = 0; i < n_states; i++) {
    for(int t = 0; t < sequence_length_minus_1; t++) {
      for(int j = 0; j < n_states; j++) {
	p_qq.set(j, i,
		 p_qq.get(j, i)
		 + p_qq_t[i].get(j, t));
      }
    }
  }
}

void ComputePq(const Matrix &p_qt,
	       Vector* p_p_q) {
  Vector &p_q = *p_p_q;
  
  int sequence_length = p_qt.n_rows();
  int n_states = p_qt.n_cols();

  p_q.Init(n_states);
  p_q.SetZero();
  for(int i = 0; i < n_states; i++) {
    for(int t = 0; t < sequence_length; t++) {
      p_q[i] += p_qt.get(t, i);
    }
  }
}

void ComputePqx(const GenMatrix<int> &sequence,
	       int n_dims,
	       const Matrix &p_qt,
	       Matrix *p_p_qx) {
  Matrix &p_qx = *p_p_qx;
  
  int n_states = p_qt.n_cols();

  p_qx.Init(n_dims, n_states);

  int sequence_length = sequence.n_cols();

  for(int i = 0; i < n_states; i++) {
    for(int a = 0; a < n_dims; a++) {
      double sum = 0;
      for(int t = 0; t < sequence_length; t++) {
	sum += p_qt.get(t, i) * (sequence.get(0, t) == a);
      }
      p_qx.set(a, i, sum);
    }
  }
}


#endif /* HMM_KERNEL_UTILS_H */
