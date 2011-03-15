struct NRsparseLinbcg : Linbcg {
	NRsparseMat &mat;
	Int n;
	NRsparseLinbcg(NRsparseMat &matrix) : mat(matrix), n(mat.nrows) {}
	void atimes(VecDoub_I &x, VecDoub_O &r, const Int itrnsp) {
		if (itrnsp) r=mat.atx(x);
		else r=mat.ax(x);
	}
	void asolve(VecDoub_I &b, VecDoub_O &x, const Int itrnsp) {
		Int i,j;
		Doub diag;
		for (i=0;i<n;i++) {
			diag=0.0;
			for (j=mat.col_ptr[i];j<mat.col_ptr[i+1];j++)
				if (mat.row_ind[j] == i) {
					diag=mat.val[j];
					break;
				}
			x[i]=(diag != 0.0 ? b[i]/diag : b[i]);
		}
	}
};
