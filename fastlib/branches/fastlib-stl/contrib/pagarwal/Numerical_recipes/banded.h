void banmul(MatDoub_I &a, const Int m1, const Int m2, VecDoub_I &x,
	VecDoub_O &b)
{
	Int i,j,k,tmploop,n=a.nrows();
	for (i=0;i<n;i++) {
		k=i-m1;
		tmploop=MIN(m1+m2+1,Int(n-k));
		b[i]=0.0;
		for (j=MAX(0,-k);j<tmploop;j++) b[i] += a[i][j]*x[j+k];
	}
}
struct Bandec {
	Int n,m1,m2;
	MatDoub au,al;
	VecInt indx;
	Doub d;
	Bandec(MatDoub_I &a, const int mm1, const int mm2);
	void solve(VecDoub_I &b, VecDoub_O &x);
	Doub det();
};
Bandec::Bandec(MatDoub_I &a, const Int mm1, const Int mm2)
	: n(a.nrows()), au(a), m1(mm1), m2(mm2), al(n,m1), indx(n)
{
	const Doub TINY=1.0e-40;
	Int i,j,k,l,mm;
	Doub dum;
	mm=m1+m2+1;
	l=m1;
	for (i=0;i<m1;i++) {
		for (j=m1-i;j<mm;j++) au[i][j-l]=au[i][j];
		l--;
		for (j=mm-l-1;j<mm;j++) au[i][j]=0.0;
	}
	d=1.0;
	l=m1;
	for (k=0;k<n;k++) {
		dum=au[k][0];
		i=k;
		if (l<n) l++;
		for (j=k+1;j<l;j++) {
			if (abs(au[j][0]) > abs(dum)) {
				dum=au[j][0];
				i=j;
			}
		}
		indx[k]=i+1;
		if (dum == 0.0) au[k][0]=TINY;
		if (i != k) {
			d = -d;
			for (j=0;j<mm;j++) SWAP(au[k][j],au[i][j]);
		}
		for (i=k+1;i<l;i++) {
			dum=au[i][0]/au[k][0];
			al[k][i-k-1]=dum;
			for (j=1;j<mm;j++) au[i][j-1]=au[i][j]-dum*au[k][j];
			au[i][mm-1]=0.0;
		}
	}
}
void Bandec::solve(VecDoub_I &b, VecDoub_O &x)
{
	Int i,j,k,l,mm;
	Doub dum;
	mm=m1+m2+1;
	l=m1;
	for (k=0;k<n;k++) x[k] = b[k];
	for (k=0;k<n;k++) {
		j=indx[k]-1;
		if (j!=k) SWAP(x[k],x[j]);
		if (l<n) l++;
		for (j=k+1;j<l;j++) x[j] -= al[k][j-k-1]*x[k];
	}
	l=1;
	for (i=n-1;i>=0;i--) {
		dum=x[i];
		for (k=1;k<l;k++) dum -= au[i][k]*x[k+i];
		x[i]=dum/au[i][0];
		if (l<mm) l++;
	}
}
Doub Bandec::det() {
	Doub dd = d;
	for (int i=0;i<n;i++) dd *= au[i][0];
	return dd;
}
