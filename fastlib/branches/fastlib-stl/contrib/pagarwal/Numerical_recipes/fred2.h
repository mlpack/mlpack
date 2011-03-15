template <class G, class K>
struct Fred2 {
	const Doub a,b;
	const Int n;
	G &g;
	K &ak;
	VecDoub t,f,w;
	Fred2(const Doub aa, const Doub bb, const Int nn, G &gg, K &akk) :
		a(aa), b(bb), n(nn), g(gg), ak(akk), t(n), f(n), w(n)
	{
		MatDoub omk(n,n);
		gauleg(a,b,t,w);
		for (Int i=0;i<n;i++) {
			for (Int j=0;j<n;j++)
				omk[i][j]=Doub(i == j)-ak(t[i],t[j])*w[j];
			f[i]=g(t[i]);
		}
		LUdcmp alu(omk);
		alu.solve(f,f);
	}

	Doub fredin(const Doub x)
	{
		Doub sum=0.0;
		for (Int i=0;i<n;i++) sum += ak(x,t[i])*w[i]*f[i];
		return g(x)+sum;
	}
};
