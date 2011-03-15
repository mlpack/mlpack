struct QRdcmp {
	Int n;
	MatDoub qt, r;
	Bool sing;
	QRdcmp(MatDoub_I &a);
	void solve(VecDoub_I &b, VecDoub_O &x);
	void qtmult(VecDoub_I &b, VecDoub_O &x);
	void rsolve(VecDoub_I &b, VecDoub_O &x);
	void update(VecDoub_I &u, VecDoub_I &v);
	void rotate(const Int i, const Doub a, const Doub b);
};
QRdcmp::QRdcmp(MatDoub_I &a)
	: n(a.nrows()), qt(n,n), r(a), sing(false) {
	Int i,j,k;
	VecDoub c(n), d(n);
	Doub scale,sigma,sum,tau;
	for (k=0;k<n-1;k++) {
		scale=0.0;
		for (i=k;i<n;i++) scale=MAX(scale,abs(r[i][k]));
		if (scale == 0.0) {
			sing=true;
			c[k]=d[k]=0.0;
		} else {
			for (i=k;i<n;i++) r[i][k] /= scale;
			for (sum=0.0,i=k;i<n;i++) sum += SQR(r[i][k]);
			sigma=SIGN(sqrt(sum),r[k][k]);
			r[k][k] += sigma;
			c[k]=sigma*r[k][k];
			d[k] = -scale*sigma;
			for (j=k+1;j<n;j++) {
				for (sum=0.0,i=k;i<n;i++) sum += r[i][k]*r[i][j];
				tau=sum/c[k];
				for (i=k;i<n;i++) r[i][j] -= tau*r[i][k];
			}
		}
	}
	d[n-1]=r[n-1][n-1];
	if (d[n-1] == 0.0) sing=true;
	for (i=0;i<n;i++) {
		for (j=0;j<n;j++) qt[i][j]=0.0;
		qt[i][i]=1.0;
	}
	for (k=0;k<n-1;k++) {
		if (c[k] != 0.0) {
			for (j=0;j<n;j++) {
				sum=0.0;
				for (i=k;i<n;i++)
					sum += r[i][k]*qt[i][j];
				sum /= c[k];
				for (i=k;i<n;i++)
					qt[i][j] -= sum*r[i][k];
			}
		}
	}
	for (i=0;i<n;i++) {
		r[i][i]=d[i];
		for (j=0;j<i;j++) r[i][j]=0.0;
	}
}
void QRdcmp::solve(VecDoub_I &b, VecDoub_O &x) {
	qtmult(b,x);
	rsolve(x,x);
}

void QRdcmp::qtmult(VecDoub_I &b, VecDoub_O &x) {
	Int i,j;
	Doub sum;
	for (i=0;i<n;i++) {
		sum = 0.;
		for (j=0;j<n;j++) sum += qt[i][j]*b[j];
		x[i] = sum;
	}
}

void QRdcmp::rsolve(VecDoub_I &b, VecDoub_O &x) {
	Int i,j;
	Doub sum;
	if (sing) throw("attempting solve in a singular QR");
	for (i=n-1;i>=0;i--) {
		sum=b[i];
		for (j=i+1;j<n;j++) sum -= r[i][j]*x[j];
		x[i]=sum/r[i][i];
	}
}
void QRdcmp::update(VecDoub_I &u, VecDoub_I &v) {
	Int i,k;
	VecDoub w(u);
	for (k=n-1;k>=0;k--)
		if (w[k] != 0.0) break;
	if (k < 0) k=0;
	for (i=k-1;i>=0;i--) {
		rotate(i,w[i],-w[i+1]);
		if (w[i] == 0.0)
			w[i]=abs(w[i+1]);
		else if (abs(w[i]) > abs(w[i+1]))
			w[i]=abs(w[i])*sqrt(1.0+SQR(w[i+1]/w[i]));
		else w[i]=abs(w[i+1])*sqrt(1.0+SQR(w[i]/w[i+1]));
	}
	for (i=0;i<n;i++) r[0][i] += w[0]*v[i];
	for (i=0;i<k;i++)
		rotate(i,r[i][i],-r[i+1][i]);
	for (i=0;i<n;i++)
		if (r[i][i] == 0.0) sing=true;
}

void QRdcmp::rotate(const Int i, const Doub a, const Doub b)
{
	Int j;
	Doub c,fact,s,w,y;
	if (a == 0.0) {
		c=0.0;
		s=(b >= 0.0 ? 1.0 : -1.0);
	} else if (abs(a) > abs(b)) {
		fact=b/a;
		c=SIGN(1.0/sqrt(1.0+(fact*fact)),a);
		s=fact*c;
	} else {
		fact=a/b;
		s=SIGN(1.0/sqrt(1.0+(fact*fact)),b);
		c=fact*s;
	}
	for (j=i;j<n;j++) {
		y=r[i][j];
		w=r[i+1][j];
		r[i][j]=c*y-s*w;
		r[i+1][j]=s*y+c*w;
	}
	for (j=0;j<n;j++) {
		y=qt[i][j];
		w=qt[i+1][j];
		qt[i][j]=c*y-s*w;
		qt[i+1][j]=s*y+c*w;
	}
}
