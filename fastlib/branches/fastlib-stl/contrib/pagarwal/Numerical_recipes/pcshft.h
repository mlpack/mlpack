void pcshft(Doub a, Doub b, VecDoub_IO &d)
{
	Int k,j,n=d.size();
	Doub cnst=2.0/(b-a), fac=cnst;
	for (j=1;j<n;j++) {
		d[j] *= fac;
		fac *= cnst;
	}
	cnst=0.5*(a+b);
	for (j=0;j<=n-2;j++)
		for (k=n-2;k>=j;k--)
			d[k] -= cnst*d[k+1];
}
void ipcshft(Doub a, Doub b, VecDoub_IO &d) {
	pcshft(-(2.+b+a)/(b-a),(2.-b-a)/(b-a),d);
}
