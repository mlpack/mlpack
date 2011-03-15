void stringalign(char *ain, char *bin, Doub mispen, Doub gappen,
	Doub skwpen, char *aout, char *bout, char *summary) {
	Int i,j,k;
	Doub dn,rt,dg;
	Int ia = strlen(ain), ib = strlen(bin);
	MatDoub cost(ia+1,ib+1);
	cost[0][0] = 0.;
	for (i=1;i<=ia;i++) cost[i][0] = cost[i-1][0] + skwpen;
	for (i=1;i<=ib;i++) cost[0][i] = cost[0][i-1] + skwpen;
	for (i=1;i<=ia;i++) for (j=1;j<=ib;j++) {
		dn = cost[i-1][j] + ((j == ib)? skwpen : gappen);
		rt = cost[i][j-1] + ((i == ia)? skwpen : gappen);
		dg = cost[i-1][j-1] + ((ain[i-1] == bin[j-1])? -1. : mispen);
		cost[i][j] = MIN(MIN(dn,rt),dg);
	}
	i=ia; j=ib; k=0;
	while (i > 0 || j > 0) {
		dn = rt = dg = 9.99e99;
		if (i>0) dn = cost[i-1][j] + ((j==ib)? skwpen : gappen);
		if (j>0) rt = cost[i][j-1] + ((i==ia)? skwpen : gappen);
		if (i>0 && j>0) dg = cost[i-1][j-1] +
			((ain[i-1] == bin[j-1])? -1. : mispen);
		if (dg <= MIN(dn,rt)) {
			aout[k] = ain[i-1];
			bout[k] = bin[j-1];
			summary[k++] = ((ain[i-1] == bin[j-1])? '=' : '!');
			i--; j--;
		}
		else if (dn < rt) {
			aout[k] = ain[i-1];
			bout[k] = ' ';
			summary[k++] = ' ';		
			i--;
		}
		else {
			aout[k] = ' ';
			bout[k] = bin[j-1];
			summary[k++] = ' ';		
			j--;
		}
	}
	for (i=0;i<k/2;i++) {
		SWAP(aout[i],aout[k-1-i]);
		SWAP(bout[i],bout[k-1-i]);
		SWAP(summary[i],summary[k-1-i]);
	}
	aout[k] = bout[k] = summary[k] = 0;
}
