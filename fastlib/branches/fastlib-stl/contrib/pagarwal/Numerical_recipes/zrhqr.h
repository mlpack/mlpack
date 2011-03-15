void zrhqr(VecDoub_I &a, VecComplex_O &rt)
{
	Int m=a.size()-1;
	MatDoub hess(m,m);
	for (Int k=0;k<m;k++) {
		hess[0][k] = -a[m-k-1]/a[m];
		for (Int j=1;j<m;j++) hess[j][k]=0.0;
		if (k != m-1) hess[k+1][k]=1.0;
	}
	Unsymmeig h(hess, false, true);
	for (Int j=0;j<m;j++)
		rt[j]=h.wri[j];
}
