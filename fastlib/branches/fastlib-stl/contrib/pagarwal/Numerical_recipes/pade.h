Ratfn pade(VecDoub_I &cof)
{
	const Doub BIG=1.0e99;
	Int j,k,n=(cof.size()-1)/2;
	Doub sum;
	MatDoub q(n,n),qlu(n,n);
	VecInt indx(n);
	VecDoub x(n),y(n),num(n+1),denom(n+1);
	for (j=0;j<n;j++) {
		y[j]=cof[n+j+1];
		for (k=0;k<n;k++) q[j][k]=cof[j-k+n];
	}
	LUdcmp lu(q);
	lu.solve(y,x);
	for (j=0;j<4;j++) lu.mprove(y,x);
	for (k=0;k<n;k++) {
		for (sum=cof[k+1],j=0;j<=k;j++) sum -= x[j]*cof[k-j];
		y[k]=sum;
	}
	num[0] = cof[0];
	denom[0] = 1.;
	for (j=0;j<n;j++) {
		num[j+1]=y[j];
		denom[j+1] = -x[j];
	}
	return Ratfn(num,denom);
}
