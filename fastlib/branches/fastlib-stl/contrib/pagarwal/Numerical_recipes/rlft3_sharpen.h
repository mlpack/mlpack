	Int i, j, nx=256, ny=256;
	MatDoub data(nx,ny);
	VecDoub speq(2*nx);
	Doub fac;
	...
	rlft3(data,speq,1);
	for (i=0;i<nx/2;i++) for (j=0;j<ny/2;j++) {
			fac = 1.+3.*sqrt(SQR(i*2./nx)+SQR(j*2./ny));
			Cmplx(data[i])[j] *= fac;
			if (i>0) Cmplx(data[nx-i])[j] *= fac;
	}
	for (j=0;j<ny/2;j++) {
		fac = 1.+3.*sqrt(1.+SQR(j*2./ny));
		Cmplx(data[nx/2])[j] *= fac;
	}
	for (i=0;i<nx/2;i++) {
		fac = 1.+3.*sqrt(SQR(i*2./nx)+1.);
		Cmplx(speq)[i] *= fac;
		if (i>0) Cmplx(speq)[nx-i] *= fac;
	}
	Cmplx(speq)[nx/2] *= (1.+3.*sqrt(2.));
	rlft3(data,speq,-1);
