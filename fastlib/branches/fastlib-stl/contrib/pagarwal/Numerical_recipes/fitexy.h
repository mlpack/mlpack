struct Fitexy{
	Doub a, b, siga, sigb, chi2, q;
	Int ndat;
	VecDoub xx,yy,sx,sy,ww;
	Doub aa, offs;

	Fitexy(VecDoub_I &x, VecDoub_I &y, VecDoub_I &sigx, VecDoub_I &sigy)
	: ndat(x.size()),xx(ndat),yy(ndat),sx(ndat),sy(ndat),ww(ndat) {
		const Doub POTN=1.571000,BIG=1.0e30,ACC=1.0e-6;
		const Doub PI=3.141592653589793238;
		Gamma gam;
		Brent brent(ACC);
		Chixy chixy(xx,yy,sx,sy,ww,aa,offs);
		Int j;
		Doub amx,amn,varx,vary,ang[7],ch[7],scale,bmn,bmx,d1,d2,r2,dum1;
		avevar(x,dum1,varx);
		avevar(y,dum1,vary);
		scale=sqrt(varx/vary);
		for (j=0;j<ndat;j++) {
			xx[j]=x[j];
			yy[j]=y[j]*scale;
			sx[j]=sigx[j];
			sy[j]=sigy[j]*scale;
			ww[j]=sqrt(SQR(sx[j])+SQR(sy[j]));
		}
		Fitab fit(xx,yy,ww);
		b = fit.b;
		offs=ang[0]=0.0;
		ang[1]=atan(b);
		ang[3]=0.0;
		ang[4]=ang[1];
		ang[5]=POTN;
		for (j=3;j<6;j++) ch[j]=chixy(ang[j]);
		brent.bracket(ang[0],ang[1],chixy);
		ang[0] = brent.ax; ang[1] = brent.bx; ang[2] = brent.cx;
		ch[0]  = brent.fa; ch[1]  = brent.fb; ch[2]  = brent.fc;
		b = brent.minimize(chixy);
		chi2=chixy(b);
		a=aa;
		q=gam.gammq(0.5*(ndat-2),chi2*0.5);
		r2=0.0;
		for (j=0;j<ndat;j++) r2 += ww[j];
		r2=1.0/r2;
		bmx=bmn=BIG;
		offs=chi2+1.0;
		for (j=0;j<6;j++) {
			if (ch[j] > offs) {
				d1=abs(ang[j]-b);
				while (d1 >= PI) d1 -= PI;
				d2=PI-d1;
				if (ang[j] < b) SWAP(d1,d2);
				if (d1 < bmx) bmx=d1;
				if (d2 < bmn) bmn=d2;
			}
		}
		if (bmx < BIG) {
			bmx=zbrent(chixy,b,b+bmx,ACC)-b;
			amx=aa-a;
			bmn=zbrent(chixy,b,b-bmn,ACC)-b;
			amn=aa-a;
			sigb=sqrt(0.5*(bmx*bmx+bmn*bmn))/(scale*SQR(cos(b)));
			siga=sqrt(0.5*(amx*amx+amn*amn)+r2)/scale;
		} else sigb=siga=BIG;
		a /= scale;
		b=tan(b)/scale;
	}

	struct Chixy {
		VecDoub &xx,&yy,&sx,&sy,&ww;
		Doub &aa,&offs;

		Chixy(VecDoub &xxx, VecDoub &yyy, VecDoub &ssx, VecDoub &ssy,
		VecDoub &www, Doub &aaa, Doub &ooffs) : xx(xxx),yy(yyy),sx(ssx),
		sy(ssy),ww(www),aa(aaa),offs(ooffs) {}

		Doub operator()(const Doub bang) {
			const Doub BIG=1.0e30;
			Int j,nn=xx.size();
			Doub ans,avex=0.0,avey=0.0,sumw=0.0,b;
			b=tan(bang);
			for (j=0;j<nn;j++) {
				ww[j] = SQR(b*sx[j])+SQR(sy[j]);
				sumw += (ww[j]=(ww[j] < 1.0/BIG ? BIG : 1.0/ww[j]));
				avex += ww[j]*xx[j];
				avey += ww[j]*yy[j];
			}
			avex /= sumw;
			avey /= sumw;
			aa=avey-b*avex;
			for (ans = -offs,j=0;j<nn;j++)
				ans += ww[j]*SQR(yy[j]-aa-b*xx[j]);
			return ans;
		}
	};

};
