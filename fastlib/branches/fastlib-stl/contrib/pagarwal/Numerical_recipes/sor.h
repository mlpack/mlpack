void sor(MatDoub_I &a, MatDoub_I &b, MatDoub_I &c, MatDoub_I &d, MatDoub_I &e,
	MatDoub_I &f, MatDoub_IO &u, const Doub rjac)
{
	const Int MAXITS=1000;
	const Doub EPS=1.0e-13;
	Doub anormf=0.0,omega=1.0;
	Int jmax=a.nrows();
	for (Int j=1;j<jmax-1;j++)
		for (Int l=1;l<jmax-1;l++)
			anormf += abs(f[j][l]);
	for (Int n=0;n<MAXITS;n++) {
		Doub anorm=0.0;
		Int jsw=1;
		for (Int ipass=0;ipass<2;ipass++) {
			Int lsw=jsw;
			for (Int j=1;j<jmax-1;j++) {
				for (Int l=lsw;l<jmax-1;l+=2) {
					Doub resid=a[j][l]*u[j+1][l]+b[j][l]*u[j-1][l]
						+c[j][l]*u[j][l+1]+d[j][l]*u[j][l-1]
						+e[j][l]*u[j][l]-f[j][l];
					anorm += abs(resid);
					u[j][l] -= omega*resid/e[j][l];
				}
				lsw=3-lsw;
			}
			jsw=3-jsw;
			omega=(n == 0 && ipass == 0 ? 1.0/(1.0-0.5*rjac*rjac) :
				1.0/(1.0-0.25*rjac*rjac*omega));
		}
		if (anorm < EPS*anormf) return;
	}
	throw("MAXITS exceeded");
}
