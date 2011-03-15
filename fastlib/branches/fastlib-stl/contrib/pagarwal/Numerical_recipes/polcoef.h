void polcoe(VecDoub_I &x, VecDoub_I &y, VecDoub_O &cof)
{
	Int k,j,i,n=x.size();
	Doub phi,ff,b;
	VecDoub s(n);
	for (i=0;i<n;i++) s[i]=cof[i]=0.0;
	s[n-1]= -x[0];
	for (i=1;i<n;i++) {
		for (j=n-1-i;j<n-1;j++)
			s[j] -= x[i]*s[j+1];
		s[n-1] -= x[i];
	}
	for (j=0;j<n;j++) {
		phi=n;
		for (k=n-1;k>0;k--)
			phi=k*s[k]+x[j]*phi;
		ff=y[j]/phi;
		b=1.0;
		for (k=n-1;k>=0;k--) {
			cof[k] += b*ff;
			b=s[k]+x[j]*b;
		}
	}
}
void polcof(VecDoub_I &xa, VecDoub_I &ya, VecDoub_O &cof)
{
	Int k,j,i,n=xa.size();
	Doub xmin;
	VecDoub x(n),y(n);
	for (j=0;j<n;j++) {
		x[j]=xa[j];
		y[j]=ya[j];
	}
	for (j=0;j<n;j++) {
		VecDoub x_t(n-j),y_t(n-j);
		for (k=0;k<n-j;k++) {
			x_t[k]=x[k];
			y_t[k]=y[k];
		}
		Poly_interp interp(x,y,n-j);
		cof[j] = interp.rawinterp(0,0.);
		xmin=1.0e99;
		k = -1;
		for (i=0;i<n-j;i++) {
			if (abs(x[i]) < xmin) {
				xmin=abs(x[i]);
				k=i;
			}
			if (x[i] != 0.0)
				y[i]=(y[i]-cof[j])/x[i];
		}
		for (i=k+1;i<n-j;i++) {
			y[i-1]=y[i];
			x[i-1]=x[i];
		}
	}
}
