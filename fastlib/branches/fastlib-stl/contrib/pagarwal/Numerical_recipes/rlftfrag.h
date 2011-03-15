Int main(void)		// example1
{
	const Int N2=256,N3=256;
	NRMat3d<Doub> data(1,N2,N3);
	MatDoub speq(1,2*N2);

//	...
	rlft3(data,speq,1);
//	...
	rlft3(data,speq,-1);
//	...
	return 0;
}

Int main(void)		// example2
{
	const Int N1=32,N2=64,N3=16;
	NRMat3d<Doub> data(N1,N2,N3);
	MatDoub speq(N1,2*N2);
//	...
	rlft3(data,speq,1);
//	...
	return 0;
}

Int main(void)		// example3
{
	Int j;
	Doub fac,r,i,*sp1,*sp2;
	const Int N=32;
	NRMat3d<Doub> data1(N,N,N),data2(N,N,N);
	MatDoub speq1(N,2*N),speq2(N,2*N);
//	...
	rlft3(data1,speq1,1);
	rlft3(data2,speq2,1);
	fac=2.0/(N*N*N);
	sp1 = &data1[0][0][0];
	sp2 = &data1[0][0][0];
	for (j=0;j<N*N*N/2;j++) {
		r = sp1[0]*sp2[0] - sp1[1]*sp2[1];
		i = sp1[0]*sp2[1] + sp1[1]*sp2[0];
		sp1[0] = fac*r;
		sp1[1] = fac*i;
		sp1 += 2;
		sp2 += 2;
	}
	sp1 = &speq1[0][0];
	sp2 = &speq2[0][0];
	for (j=0;j<N*N;j++) {
		r = sp1[0]*sp2[0] - sp1[1]*sp2[1];
		i = sp1[0]*sp2[1] + sp1[1]*sp2[0];
		sp1[0] = fac*r;
		sp1[1] = fac*i;
		sp1 += 2;
		sp2 += 2;
	}
	rlft3(data1,speq1,-1);
//	...
	return 0;
}
