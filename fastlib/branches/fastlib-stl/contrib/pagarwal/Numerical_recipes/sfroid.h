Int main_sfroid(void)
{
	const Int M=40,MM=4;
	const Int NE=3,NB=1,NYJ=NE,NYK=M+1;
	Int mm=3,n=5,mpt=M+1;
	VecInt indexv(NE);
	VecDoub x(M+1),scalv(NE);
	MatDoub y(NYJ,NYK);
	Int itmax=100;
	Doub c2[]={16.0,20.0,-16.0,-20.0};
	Doub conv=1.0e-14,slowc=1.0,h=1.0/M;
	if ((n+mm & 1) != 0) {
		indexv[0]=0;
		indexv[1]=1;
		indexv[2]=2;
	} else {
		indexv[0]=1;
		indexv[1]=0;
		indexv[2]=2;
	}
	Doub anorm=1.0;
	if (mm != 0) {
		Doub q1=n;
		for (Int i=1;i<=mm;i++) anorm = -0.5*anorm*(n+i)*(q1--/i);
	}
	for (Int k=0;k<M;k++) {
		x[k]=k*h;
		Doub fac1=1.0-x[k]*x[k];
		Doub fac2=exp((-mm/2.0)*log(fac1));
		y[0][k]=plgndr(n,mm,x[k])*fac2;
		Doub deriv = -((n-mm+1)*plgndr(n+1,mm,x[k])-
			(n+1)*x[k]*plgndr(n,mm,x[k]))/fac1;
		y[1][k]=mm*x[k]*y[0][k]/fac1+deriv*fac2;
		y[2][k]=n*(n+1)-mm*(mm+1);
	}
	x[M]=1.0;
	y[0][M]=anorm;
	y[2][M]=n*(n+1)-mm*(mm+1);
	y[1][M]=y[2][M]*y[0][M]/(2.0*(mm+1.0));
	scalv[0]=abs(anorm);
	scalv[1]=(y[1][M] > scalv[0] ? y[1][M] : scalv[0]);
	scalv[2]=(y[2][M] > 1.0 ? y[2][M] : 1.0);
	for (Int j=0;j<MM;j++) {
		Difeq difeq(mm,n,mpt,h,c2[j],anorm,x);
		Solvde solvde(itmax,conv,slowc,scalv,indexv,NB,y,difeq);
		cout << endl << " m = " << setw(3) << mm;
		cout << "  n = " << setw(3) << n << "  c**2 = ";
		cout << fixed << setprecision(3) << setw(7) << c2[j];
		cout << " lamda = " << setprecision(6) << (y[2][0]+mm*(mm+1));
		cout << endl;
	}
	return 0;
}
