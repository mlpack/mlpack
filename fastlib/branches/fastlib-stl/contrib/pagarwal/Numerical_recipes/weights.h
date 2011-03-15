void weights(const Doub z, VecDoub_I &x, MatDoub_O &c)
{
	Int n=c.nrows()-1;
	Int m=c.ncols()-1;
	Doub c1=1.0;
	Doub c4=x[0]-z;
	for (Int k=0;k<=m;k++)
		for (Int j=0;j<=n;j++)
			c[j][k]=0.0;
	c[0][0]=1.0;
	for (Int i=1;i<=n;i++) {
		Int mn=MIN(i,m);
		Doub c2=1.0;
		Doub c5=c4;
		c4=x[i]-z;
		for (Int j=0;j<i;j++) {
			Doub c3=x[i]-x[j];
			c2=c2*c3;
			if (j == i-1) {
				for (Int k=mn;k>0;k--)
					c[i][k]=c1*(k*c[i-1][k-1]-c5*c[i-1][k])/c2;
				c[i][0]=-c1*c5*c[i-1][0]/c2;
			}
			for (Int k=mn;k>0;k--)
				c[j][k]=(c4*c[j][k]-k*c[j][k-1])/c3;
			c[j][0]=c4*c[j][0]/c3;
		}
		c1=c2;
	}
}
