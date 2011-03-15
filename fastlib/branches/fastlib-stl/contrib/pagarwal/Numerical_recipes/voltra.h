template <class G, class K>
void voltra(const Doub t0, const Doub h, G &g, K &ak, VecDoub_O &t, MatDoub_O &f)
{
	Int m=f.nrows();
	Int n=f.ncols();
	VecDoub b(m);
	MatDoub a(m,m);
	t[0]=t0;
	for (Int k=0;k<m;k++) f[k][0]=g(k,t[0]);
	for (Int i=1;i<n;i++) {
		t[i]=t[i-1]+h;
		for (Int k=0;k<m;k++) {
			Doub sum=g(k,t[i]);
			for (Int l=0;l<m;l++) {
				sum += 0.5*h*ak(k,l,t[i],t[0])*f[l][0];
				for (Int j=1;j<i;j++)
					sum += h*ak(k,l,t[i],t[j])*f[l][j];
				if (k == l)
					a[k][l]=1.0-0.5*h*ak(k,l,t[i],t[i]);
				else
					a[k][l] = -0.5*h*ak(k,l,t[i],t[i]);
			}
			b[k]=sum;
		}
		LUdcmp alu(a);
		alu.solve(b,b);
		for (Int k=0;k<m;k++) f[k][i]=b[k];
	}
}
