struct Solvde {
	const Int itmax;
	const Doub conv;
	const Doub slowc;
	const VecDoub &scalv;
	const VecInt &indexv;
	const Int nb;
	MatDoub &y;
	Difeq &difeq;
	Int ne,m;
	VecInt kmax;
	VecDoub ermax;
	Mat3DDoub c;
	MatDoub s;
	Solvde(const Int itmaxx, const Doub convv, const Doub slowcc,
		VecDoub_I &scalvv, VecInt_I &indexvv, const Int nbb,
		MatDoub_IO &yy, Difeq &difeqq);
	void pinvs(const Int ie1, const Int ie2, const Int je1, const Int jsf,
		const Int jc1, const Int k);
	void bksub(const Int ne, const Int nb, const Int jf, const Int k1,
		const Int k2);
	void red(const Int iz1, const Int iz2, const Int jz1, const Int jz2,
		const Int jm1, const Int jm2, const Int jmf, const Int ic1,
		const Int jc1, const Int jcf, const Int kc);
};

Solvde::Solvde(const Int itmaxx, const Doub convv, const Doub slowcc,
	VecDoub_I &scalvv, VecInt_I &indexvv, const Int nbb, MatDoub_IO &yy,
	Difeq &difeqq) : itmax(itmaxx), conv(convv), slowc(slowcc),
	scalv(scalvv), indexv(indexvv), nb(nbb), y(yy), difeq(difeqq),
	ne(y.nrows()), m(y.ncols()), kmax(ne), ermax(ne), c(ne,ne-nb+1,m+1),
	s(ne,2*ne+1)
{
	Int jv,k,nvars=ne*m;
	Int k1=0,k2=m;
	Int j1=0,j2=nb,j3=nb,j4=ne,j5=j4+j1,j6=j4+j2,j7=j4+j3,j8=j4+j4,j9=j8+j1;
	Int ic1=0,ic2=ne-nb,ic3=ic2,ic4=ne,jc1=0,jcf=ic3;
	for (Int it=0;it<itmax;it++) {
		k=k1;
		difeq.smatrix(k,k1,k2,j9,ic3,ic4,indexv,s,y);
		pinvs(ic3,ic4,j5,j9,jc1,k1);
		for (k=k1+1;k<k2;k++) {
			Int kp=k;
			difeq.smatrix(k,k1,k2,j9,ic1,ic4,indexv,s,y);
			red(ic1,ic4,j1,j2,j3,j4,j9,ic3,jc1,jcf,kp);
			pinvs(ic1,ic4,j3,j9,jc1,k);
		}
		k=k2;
		difeq.smatrix(k,k1,k2,j9,ic1,ic2,indexv,s,y);
		red(ic1,ic2,j5,j6,j7,j8,j9,ic3,jc1,jcf,k2);
		pinvs(ic1,ic2,j7,j9,jcf,k2);
		bksub(ne,nb,jcf,k1,k2);
		Doub err=0.0;
		for (Int j=0;j<ne;j++) {
			jv=indexv[j];
			Doub errj=0.0,vmax=0.0;
			Int km=0;
			for (k=k1;k<k2;k++) {
				Doub vz=abs(c[jv][0][k]);
				if (vz > vmax) {
					vmax=vz;
					km=k+1;
				}
				errj += vz;
			}
			err += errj/scalv[j];
			ermax[j]=c[jv][0][km-1]/scalv[j];
			kmax[j]=km;
		}
		err /= nvars;
		Doub fac=(err > slowc ? slowc/err : 1.0);
		for (Int j=0;j<ne;j++) {
			jv=indexv[j];
			for (k=k1;k<k2;k++)
			y[j][k] -= fac*c[jv][0][k];
		}
		cout << setw(8) << "Iter.";
		cout << setw(10) << "Error" << setw(10) <<  "FAC" << endl;
		cout << setw(6) << it;
		cout << fixed << setprecision(6) << setw(13) << err;
		cout << setw(12) << fac << endl;
		if (err < conv) return;
	}
	throw("Too many iterations in solvde");
}

void Solvde::pinvs(const Int ie1, const Int ie2, const Int je1, const Int jsf,
	const Int jc1, const Int k)
{
	Int jpiv,jp,je2,jcoff,j,irow,ipiv,id,icoff,i;
	Doub pivinv,piv,big;
	const Int iesize=ie2-ie1;
	VecInt indxr(iesize);
	VecDoub pscl(iesize);
	je2=je1+iesize;
	for (i=ie1;i<ie2;i++) {
		big=0.0;
		for (j=je1;j<je2;j++)
			if (abs(s[i][j]) > big) big=abs(s[i][j]);
		if (big == 0.0)
			throw("Singular matrix - row all 0, in pinvs");
		pscl[i-ie1]=1.0/big;
		indxr[i-ie1]=0;
	}
	for (id=0;id<iesize;id++) {
		piv=0.0;
		for (i=ie1;i<ie2;i++) {
			if (indxr[i-ie1] == 0) {
				big=0.0;
				for (j=je1;j<je2;j++) {
					if (abs(s[i][j]) > big) {
						jp=j;
						big=abs(s[i][j]);
					}
				}
				if (big*pscl[i-ie1] > piv) {
					ipiv=i;
					jpiv=jp;
					piv=big*pscl[i-ie1];
				}
			}
		}
		if (s[ipiv][jpiv] == 0.0)
			throw("Singular matrix in routine pinvs");
		indxr[ipiv-ie1]=jpiv+1;
		pivinv=1.0/s[ipiv][jpiv];
		for (j=je1;j<=jsf;j++) s[ipiv][j] *= pivinv;
		s[ipiv][jpiv]=1.0;
		for (i=ie1;i<ie2;i++) {
			if (indxr[i-ie1] != jpiv+1) {
				if (s[i][jpiv] != 0.0) {
					Doub dum=s[i][jpiv];
					for (j=je1;j<=jsf;j++)
						s[i][j] -= dum*s[ipiv][j];
					s[i][jpiv]=0.0;
				}
			}
		}
	}
	jcoff=jc1-je2;
	icoff=ie1-je1;
	for (i=ie1;i<ie2;i++) {
		irow=indxr[i-ie1]+icoff;
		for (j=je2;j<=jsf;j++) c[irow-1][j+jcoff][k]=s[i][j];
	}
}

void Solvde::bksub(const Int ne, const Int nb, const Int jf, const Int k1,
	const Int k2)
{
	Int nbf=ne-nb,im=1;
	for (Int k=k2-1;k>=k1;k--) {
		if (k == k1) im=nbf+1;
		Int kp=k+1;
		for (Int j=0;j<nbf;j++) {
			Doub xx=c[j][jf][kp];
			for (Int i=im-1;i<ne;i++)
				c[i][jf][k] -= c[i][j][k]*xx;
		}
	}
	for (Int k=k1;k<k2;k++) {
		Int kp=k+1;
		for (Int i=0;i<nb;i++) c[i][0][k]=c[i+nbf][jf][k];
		for (Int i=0;i<nbf;i++) c[i+nb][0][k]=c[i][jf][kp];
	}
}

void Solvde::red(const Int iz1, const Int iz2, const Int jz1, const Int jz2,
	const Int jm1, const Int jm2, const Int jmf, const Int ic1,
	const Int jc1, const Int jcf, const Int kc)
{
	Int l,j,i;
	Doub vx;
	Int loff=jc1-jm1,ic=ic1;
	for (j=jz1;j<jz2;j++) {
		for (l=jm1;l<jm2;l++) {
			vx=c[ic][l+loff][kc-1];
			for (i=iz1;i<iz2;i++) s[i][l] -= s[i][j]*vx;
		}
		vx=c[ic][jcf][kc-1];
		for (i=iz1;i<iz2;i++) s[i][jmf] -= s[i][j]*vx;
		ic += 1;
	}
}
