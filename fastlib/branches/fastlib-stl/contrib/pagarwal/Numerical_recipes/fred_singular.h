template <class Q>
struct Wwghts {
	Doub h;
	Int n;
	Q &quad;
	VecDoub wghts;
	Wwghts(Doub hh, Int nn, Q &q) : h(hh), n(nn), quad(q), wghts(n) {}
	VecDoub weights()
	{
		Int k;
		Doub fac;
		Doub hi=1.0/h;
		for (Int j=0;j<n;j++)
			wghts[j]=0.0;
		if (n >= 4) {
			VecDoub wold(4),wnew(4),w(4);
			wold=quad.kermom(0.0);
			Doub b=0.0;
			for (Int j=0;j<n-3;j++) {
				Doub c=j;
				Doub a=b;
				b=a+h;
				if (j == n-4) b=(n-1)*h;
				wnew=quad.kermom(b);
				for (fac=1.0,k=0;k<4;k++,fac*=hi)
					w[k]=(wnew[k]-wold[k])*fac;
				wghts[j] += (((c+1.0)*(c+2.0)*(c+3.0)*w[0]
					-(11.0+c*(12.0+c*3.0))*w[1]+3.0*(c+2.0)*w[2]-w[3])/6.0);
				wghts[j+1] += ((-c*(c+2.0)*(c+3.0)*w[0]
					+(6.0+c*(10.0+c*3.0))*w[1]-(3.0*c+5.0)*w[2]+w[3])*0.5);
				wghts[j+2] += ((c*(c+1.0)*(c+3.0)*w[0]
					-(3.0+c*(8.0+c*3.0))*w[1]+(3.0*c+4.0)*w[2]-w[3])*0.5);
				wghts[j+3] += ((-c*(c+1.0)*(c+2.0)*w[0]
					+(2.0+c*(6.0+c*3.0))*w[1]-3.0*(c+1.0)*w[2]+w[3])/6.0);
				for (k=0;k<4;k++) wold[k]=wnew[k];
			}
		} else if (n == 3) {
			VecDoub wold(3),wnew(3),w(3);
			wold=quad.kermom(0.0);
			wnew=quad.kermom(h+h);
			w[0]=wnew[0]-wold[0];
			w[1]=hi*(wnew[1]-wold[1]);
			w[2]=hi*hi*(wnew[2]-wold[2]);
			wghts[0]=w[0]-1.5*w[1]+0.5*w[2];
			wghts[1]=2.0*w[1]-w[2];
			wghts[2]=0.5*(w[2]-w[1]);
		} else if (n == 2) {
			VecDoub wold(2),wnew(2),w(2);
			wold=quad.kermom(0.0);
			wnew=quad.kermom(h);
			wghts[0]=wnew[0]-wold[0]-(wghts[1]=hi*(wnew[1]-wold[1]));
		}
		return wghts;
	}
};
struct Quad_matrix {
	Int n;
	Doub x;
	Quad_matrix(MatDoub_O &a) : n(a.nrows())
	{
		const Doub PI=3.14159263589793238;
		VecDoub wt(n);
		Doub h=PI/(n-1);
		Wwghts<Quad_matrix> w(h,n,*this);
		for (Int j=0;j<n;j++) {
			x=j*h;
			wt=w.weights();
			Doub cx=cos(x);
			for (Int k=0;k<n;k++)
				a[j][k]=wt[k]*cx*cos(k*h);
			++a[j][j];
		}
	}
	VecDoub kermom(const Doub y)
	{
		Doub d,df,clog,x2,x3,x4,y2;
		VecDoub w(4);
		if (y >= x) {
			d=y-x;
			df=2.0*sqrt(d)*d;
			w[0]=df/3.0;
			w[1]=df*(x/3.0+d/5.0);
			w[2]=df*((x/3.0 + 0.4*d)*x + d*d/7.0);
			w[3]=df*(((x/3.0 + 0.6*d)*x + 3.0*d*d/7.0)*x+d*d*d/9.0);
		} else {
			x3=(x2=x*x)*x;
			x4=x2*x2;
			y2=y*y;
			d=x-y;
			w[0]=d*((clog=log(d))-1.0);
			w[1] = -0.25*(3.0*x+y-2.0*clog*(x+y))*d;
			w[2]=(-11.0*x3+y*(6.0*x2+y*(3.0*x+2.0*y))
				+6.0*clog*(x3-y*y2))/18.0;
			w[3]=(-25.0*x4+y*(12.0*x3+y*(6.0*x2+y*
				(4.0*x+3.0*y)))+12.0*clog*(x4-(y2*y2)))/48.0;
		}
		return w;
	}
};
Int main_fredex(void)
{
	const Int N=40;
	const Doub PI=3.141592653589793238;
	VecDoub g(N);
	MatDoub a(N,N);
	Quad_matrix qmx(a);
	LUdcmp alu(a);
	for (Int j=0;j<N;j++)
		g[j]=sin(j*PI/(N-1));
	alu.solve(g,g);
	for (Int j=0;j<N;j++) {
		Doub x=j*PI/(N-1);
		cout << fixed << setprecision(2) << setw(6) << (j+1);
		cout << setprecision(6) << setw(13) << x << setw(13) << g[j] << endl;
	}
	return 0;
}
