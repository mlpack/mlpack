void rebin(const Doub rc, const Int nd, VecDoub_I &r, VecDoub_O &xin,
	MatDoub_IO &xi, const Int j) {
	Int i,k=0;
	Doub dr=0.0,xn=0.0,xo=0.0;

	for (i=0;i<nd-1;i++) {
		while (rc > dr)
			dr += r[(++k)-1];
		if (k > 1) xo=xi[j][k-2];
		xn=xi[j][k-1];
		dr -= rc;
		xin[i]=xn-(xn-xo)*dr/r[k-1];
	}
	for (i=0;i<nd-1;i++) xi[j][i]=xin[i];
	xi[j][nd-1]=1.0;
}
