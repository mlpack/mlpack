void savgol(VecDoub_O &c, const Int np, const Int nl, const Int nr,
	const Int ld, const Int m)
{
	Int j,k,imj,ipj,kk,mm;
	Doub fac,sum;
	if (np < nl+nr+1 || nl < 0 || nr < 0 || ld > m || nl+nr < m)
		throw("bad args in savgol");
	VecInt indx(m+1);
	MatDoub a(m+1,m+1);
	VecDoub b(m+1);
	for (ipj=0;ipj<=(m << 1);ipj++) {
		sum=(ipj ? 0.0 : 1.0);
		for (k=1;k<=nr;k++) sum += pow(Doub(k),Doub(ipj));
		for (k=1;k<=nl;k++) sum += pow(Doub(-k),Doub(ipj));
		mm=MIN(ipj,2*m-ipj);
		for (imj = -mm;imj<=mm;imj+=2) a[(ipj+imj)/2][(ipj-imj)/2]=sum;
	}
	LUdcmp alud(a);
	for (j=0;j<m+1;j++) b[j]=0.0;
	b[ld]=1.0;
	alud.solve(b,b);
	for (kk=0;kk<np;kk++) c[kk]=0.0;
	for (k = -nl;k<=nr;k++) {
		sum=b[0];
		fac=1.0;
		for (mm=1;mm<=m;mm++) sum += b[mm]*(fac *= k);
		kk=(np-k) % np;
		c[kk]=sum;
	}
}
