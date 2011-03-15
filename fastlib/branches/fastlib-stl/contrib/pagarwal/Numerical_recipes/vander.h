void vander(VecDoub_I &x, VecDoub_O &w, VecDoub_I &q)
{
	Int i,j,k,n=q.size();
	Doub b,s,t,xx;
	VecDoub c(n);
	if (n == 1) w[0]=q[0];
	else {
		for (i=0;i<n;i++) c[i]=0.0;
		c[n-1] = -x[0];
		for (i=1;i<n;i++) {
			xx = -x[i];
			for (j=(n-1-i);j<(n-1);j++) c[j] += xx*c[j+1];
			c[n-1] += xx;
		}
		for (i=0;i<n;i++) {
			xx=x[i];
			t=b=1.0;
			s=q[n-1];
			for (k=n-1;k>0;k--) {
				b=c[k]+xx*b;
				s += q[k-1]*b;
				t=xx*t+b;
			}
			w[i]=s/t;
		}
	}
}
