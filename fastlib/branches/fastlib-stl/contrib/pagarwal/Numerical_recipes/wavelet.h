struct Wavelet {
	virtual void filt(VecDoub_IO &a, const Int n, const Int isign) = 0;
	virtual void condition(VecDoub_IO &a, const Int n, const Int isign) {}
};

struct Daub4 : Wavelet {
	void filt(VecDoub_IO &a, const Int n, const Int isign) {
		const Doub C0=0.4829629131445341, C1=0.8365163037378077,
		C2=0.2241438680420134, C3=-0.1294095225512603;
		Int nh,i,j;
		if (n < 4) return;
		VecDoub wksp(n);
		nh = n >> 1;
		if (isign >= 0) {
			for (i=0,j=0;j<n-3;j+=2,i++) {
				wksp[i] = C0*a[j]+C1*a[j+1]+C2*a[j+2]+C3*a[j+3];
				wksp[i+nh] = C3*a[j]-C2*a[j+1]+C1*a[j+2]-C0*a[j+3];
			}
			wksp[i] = C0*a[n-2]+C1*a[n-1]+C2*a[0]+C3*a[1];
			wksp[i+nh] = C3*a[n-2]-C2*a[n-1]+C1*a[0]-C0*a[1];
		} else {
			wksp[0] = C2*a[nh-1]+C1*a[n-1]+C0*a[0]+C3*a[nh];
			wksp[1] = C3*a[nh-1]-C0*a[n-1]+C1*a[0]-C2*a[nh];
			for (i=0,j=2;i<nh-1;i++) {
				wksp[j++] = C2*a[i]+C1*a[i+nh]+C0*a[i+1]+C3*a[i+nh+1];
				wksp[j++] = C3*a[i]-C0*a[i+nh]+C1*a[i+1]-C2*a[i+nh+1];
			}
		}
		for (i=0;i<n;i++) a[i]=wksp[i];
	}
};
void wt1(VecDoub_IO &a, const Int isign, Wavelet &wlet)
{
	Int nn, n=a.size();
	if (n < 4) return;
	if (isign >= 0) {
		wlet.condition(a,n,1);
		for (nn=n;nn>=4;nn>>=1) wlet.filt(a,nn,isign);
	} else {
		for (nn=4;nn<=n;nn<<=1) wlet.filt(a,nn,isign);
		wlet.condition(a,n,-1);
	}
}
struct Daubs : Wavelet {
	Int ncof,ioff,joff;
	VecDoub cc,cr;
	static Doub c4[4],c12[12],c20[20];
	Daubs(Int n) : ncof(n), cc(n), cr(n) {
		Int i;
		ioff = joff = -(n >> 1);
		// ioff = -2; joff = -n + 2;
		if (n == 4) for (i=0; i<n; i++) cc[i] = c4[i];
		else if (n == 12) for (i=0; i<n; i++) cc[i] = c12[i];
		else if (n == 20) for (i=0; i<n; i++) cc[i] = c20[i];
		else throw("n not yet implemented in Daubs");
		Doub sig = -1.0;
		for (i=0; i<n; i++) {
			cr[n-1-i]=sig*cc[i];
			sig = -sig;
		}
	}
	void filt(VecDoub_IO &a, const Int n, const Int isign);
};

Doub Daubs::c4[4]=
	{0.4829629131445341,0.8365163037378079,
	0.2241438680420134,-0.1294095225512604};
Doub Daubs::c12[12]=
	{0.111540743350, 0.494623890398, 0.751133908021,
	0.315250351709,-0.226264693965,-0.129766867567,
	0.097501605587, 0.027522865530,-0.031582039318,
	0.000553842201, 0.004777257511,-0.001077301085};
Doub Daubs::c20[20]=
	{0.026670057901, 0.188176800078, 0.527201188932,
	0.688459039454, 0.281172343661,-0.249846424327,
	-0.195946274377, 0.127369340336, 0.093057364604,
	-0.071394147166,-0.029457536822, 0.033212674059,
	0.003606553567,-0.010733175483, 0.001395351747,
	0.001992405295,-0.000685856695,-0.000116466855,
	0.000093588670,-0.000013264203};
void Daubs::filt(VecDoub_IO &a, const Int n, const Int isign) {
	Doub ai,ai1;
	Int i,ii,j,jf,jr,k,n1,ni,nj,nh,nmod;
	if (n < 4) return;
	VecDoub wksp(n);
	nmod = ncof*n;
	n1 = n-1;
	nh = n >> 1;
	for (j=0;j<n;j++) wksp[j]=0.0;
	if (isign >= 0) {
		for (ii=0,i=0;i<n;i+=2,ii++) {
			ni = i+1+nmod+ioff;
			nj = i+1+nmod+joff;
			for (k=0;k<ncof;k++) {
				jf = n1 & (ni+k+1);
				jr = n1 & (nj+k+1);
				wksp[ii] += cc[k]*a[jf];
				wksp[ii+nh] += cr[k]*a[jr];
			}
		}
	} else {
		for (ii=0,i=0;i<n;i+=2,ii++) {
			ai = a[ii];
			ai1 = a[ii+nh];
			ni = i+1+nmod+ioff;
			nj = i+1+nmod+joff;
			for (k=0;k<ncof;k++) {
				jf = n1 & (ni+k+1);
				jr = n1 & (nj+k+1);
				wksp[jf] += cc[k]*ai;
				wksp[jr] += cr[k]*ai1;
			}
		}
	}
	for (j=0;j<n;j++) a[j] = wksp[j];
}
struct Daub4i : Wavelet {
	void filt(VecDoub_IO &a, const Int n, const Int isign) {
		const Doub C0=0.4829629131445341, C1=0.8365163037378077,
			C2=0.2241438680420134, C3=-0.1294095225512603;
		const Doub R00=0.603332511928053,R01=0.690895531839104,
			R02=-0.398312997698228,R10=-0.796543516912183,R11=0.546392713959015,
			R12=-0.258792248333818,R20=0.0375174604524466,R21=0.457327659851769,
			R22=0.850088102549165,R23=0.223820356983114,R24=-0.129222743354319,
			R30=0.0100372245644139,R31=0.122351043116799,R32=0.227428111655837,
			R33=-0.836602921223654,R34=0.483012921773304,R43=0.443149049637559,
			R44=0.767556669298114,R45=0.374955331645687,R46=0.190151418429955,
			R47=-0.194233407427412,R53=0.231557595006790,R54=0.401069519430217,
			R55=-0.717579999353722,R56=-0.363906959570891,R57=0.371718966535296,
			R65=0.230389043796969,R66=0.434896997965703,R67=0.870508753349866,
			R75=-0.539822500731772,R76=0.801422961990337,R77=-0.257512919478482;
		Int nh,i,j;
		if (n < 8) return;
		VecDoub wksp(n);
		nh = n >> 1;
		if (isign >= 0) {
			wksp[0]  = R00*a[0]+R01*a[1]+R02*a[2];
			wksp[nh] = R10*a[0]+R11*a[1]+R12*a[2];
			wksp[1] = R20*a[0]+R21*a[1]+R22*a[2]+R23*a[3]+R24*a[4];
			wksp[nh+1] = R30*a[0]+R31*a[1]+R32*a[2]+R33*a[3]+R34*a[4];
			for (i=2,j=3;j<n-4;j+=2,i++) {
				wksp[i] = C0*a[j]+C1*a[j+1]+C2*a[j+2]+C3*a[j+3];
				wksp[i+nh] = C3*a[j]-C2*a[j+1]+C1*a[j+2]-C0*a[j+3];
			}
			wksp[nh-2] = R43*a[n-5]+R44*a[n-4]+R45*a[n-3]+R46*a[n-2]+R47*a[n-1];
			wksp[n-2] = R53*a[n-5]+R54*a[n-4]+R55*a[n-3]+R56*a[n-2]+R57*a[n-1];
			wksp[nh-1] = R65*a[n-3]+R66*a[n-2]+R67*a[n-1];
			wksp[n-1] = R75*a[n-3]+R76*a[n-2]+R77*a[n-1];
		} else {
			wksp[0] = R00*a[0]+R10*a[nh]+R20*a[1]+R30*a[nh+1];
			wksp[1] = R01*a[0]+R11*a[nh]+R21*a[1]+R31*a[nh+1];
			wksp[2] = R02*a[0]+R12*a[nh]+R22*a[1]+R32*a[nh+1];
			if (n == 8) {
				wksp[3] = R23*a[1]+R33*a[5]+R43*a[2]+R53*a[6];
				wksp[4] = R24*a[1]+R34*a[5]+R44*a[2]+R54*a[6];
			} else {
				wksp[3] = R23*a[1]+R33*a[nh+1]+C0*a[2]+C3*a[nh+2];
				wksp[4] = R24*a[1]+R34*a[nh+1]+C1*a[2]-C2*a[nh+2];
				wksp[n-5] = C2*a[nh-3]+C1*a[n-3]+R43*a[nh-2]+R53*a[n-2];
				wksp[n-4] = C3*a[nh-3]-C0*a[n-3]+R44*a[nh-2]+R54*a[n-2];
			}
			for (i=2,j=5;i<nh-3;i++) {
				wksp[j++] = C2*a[i]+C1*a[i+nh]+C0*a[i+1]+C3*a[i+nh+1];
				wksp[j++] = C3*a[i]-C0*a[i+nh]+C1*a[i+1]-C2*a[i+nh+1];
			}
			wksp[n-3] = R45*a[nh-2]+R55*a[n-2]+R65*a[nh-1]+R75*a[n-1];
			wksp[n-2] = R46*a[nh-2]+R56*a[n-2]+R66*a[nh-1]+R76*a[n-1];
			wksp[n-1] = R47*a[nh-2]+R57*a[n-2]+R67*a[nh-1]+R77*a[n-1];
		}
		for (i=0;i<n;i++) a[i]=wksp[i];
	}
	void condition(VecDoub_IO &a, const Int n, const Int isign) {
		Doub t0,t1,t2,t3;
		if (n < 4) return;
		if (isign >= 0) {
			t0 = 0.324894048898962*a[0]+0.0371580151158803*a[1];
			t1 = 1.00144540498130*a[1];
			t2 = 1.08984305289504*a[n-2];
			t3 = -0.800813234246437*a[n-2]+2.09629288435324*a[n-1];
			a[0]=t0; a[1]=t1; a[n-2]=t2; a[n-1]=t3;
		} else {
			t0 = 3.07792649138669*a[0]-0.114204567242137*a[1];
			t1 = 0.998556681198888*a[1];
			t2 = 0.917563310922261*a[n-2];
			t3 = 0.350522032550918*a[n-2]+0.477032578540915*a[n-1];
			a[0]=t0; a[1]=t1; a[n-2]=t2; a[n-1]=t3;
		}
	}
};
void wtn(VecDoub_IO &a, VecInt_I &nn, const Int isign, Wavelet &wlet)
{
	Int idim,i1,i2,i3,k,n,nnew,nprev=1,nt,ntot=1;
	Int ndim=nn.size();
	for (idim=0;idim<ndim;idim++) ntot *= nn[idim];
	if (ntot&(ntot-1)) throw("all lengths must be powers of 2 in wtn");
	for (idim=0;idim<ndim;idim++) {
		n=nn[idim];
		VecDoub wksp(n);
		nnew=n*nprev;
		if (n > 4) {
			for (i2=0;i2<ntot;i2+=nnew) {
				for (i1=0;i1<nprev;i1++) {
					for (i3=i1+i2,k=0;k<n;k++,i3+=nprev) wksp[k]=a[i3];
					if (isign >= 0) {
						wlet.condition(wksp,n,1);
						for(nt=n;nt>=4;nt >>= 1) wlet.filt(wksp,nt,isign);
					} else {
						for(nt=4;nt<=n;nt <<= 1) wlet.filt(wksp,nt,isign);
						wlet.condition(wksp,n,-1);
					}
					for (i3=i1+i2,k=0;k<n;k++,i3+=nprev) a[i3]=wksp[k];
				}
			}
		}
		nprev=nnew;
	}
}
