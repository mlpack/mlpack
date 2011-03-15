void eclass(VecInt_O &nf, VecInt_I &lista, VecInt_I &listb)
{
	Int l,k,j,n=nf.size(),m=lista.size();
	for (k=0;k<n;k++) nf[k]=k;
	for (l=0;l<m;l++) {
		j=lista[l];
		while (nf[j] != j) j=nf[j];
		k=listb[l];
		while (nf[k] != k) k=nf[k];
		if (j != k) nf[j]=k;
	}
	for (j=0;j<n;j++)
		while (nf[j] != nf[nf[j]]) nf[j]=nf[nf[j]];
}
void eclazz(VecInt_O &nf, Bool equiv(const Int, const Int))
{
	Int kk,jj,n=nf.size();
	nf[0]=0;
	for (jj=1;jj<n;jj++) {
		nf[jj]=jj;
		for (kk=0;kk<jj;kk++) {
			nf[kk]=nf[nf[kk]];
			if (equiv(jj+1,kk+1)) nf[nf[nf[kk]]]=jj;
		}
	}
	for (jj=0;jj<n;jj++) nf[jj]=nf[nf[jj]];
}
