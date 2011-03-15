void four1(Doub *data, const Int n, const Int isign) {
	Int nn,mmax,m,j,istep,i;
	Doub wtemp,wr,wpr,wpi,wi,theta,tempr,tempi;
	if (n<2 || n&(n-1)) throw("n must be power of 2 in four1");
	nn = n << 1;
	j = 1;
	for (i=1;i<nn;i+=2) {
		if (j > i) {
			SWAP(data[j-1],data[i-1]);
			SWAP(data[j],data[i]);
		}
		m=n;
		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}
	mmax=2;
	while (nn > mmax) {
		istep=mmax << 1;
		theta=isign*(6.28318530717959/mmax);
		wtemp=sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi=sin(theta);
		wr=1.0;
		wi=0.0;
		for (m=1;m<mmax;m+=2) {
			for (i=m;i<=nn;i+=istep) {
				j=i+mmax;
				tempr=wr*data[j-1]-wi*data[j];
				tempi=wr*data[j]+wi*data[j-1];
				data[j-1]=data[i-1]-tempr;
				data[j]=data[i]-tempi;
				data[i-1] += tempr;
				data[i] += tempi;
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
		mmax=istep;
	}
}
void four1(VecDoub_IO &data, const Int isign) {
	four1(&data[0],data.size()/2,isign);
}

void four1(VecComplex_IO &data, const Int isign) {
	four1((Doub*)(&data[0]),data.size(),isign);
}
struct WrapVecDoub {
	VecDoub vvec;
	VecDoub &v;
	Int n, mask;

	WrapVecDoub(const Int nn) : vvec(nn), v(vvec), n(nn/2),
	mask(n-1) {validate();}

	WrapVecDoub(VecDoub &vec) : v(vec), n(vec.size()/2),
	mask(n-1) {validate();}
		
	void validate() {if (n&(n-1)) throw("vec size must be power of 2");}

	inline Complex& operator[] (Int i) {return (Complex &)v[(i&mask) << 1];}

	inline Doub& real(Int i) {return v[(i&mask) << 1];}

	inline Doub& imag(Int i) {return v[((i&mask) << 1)+1];}

	operator VecDoub&() {return v;}

};
void realft(VecDoub_IO &data, const Int isign) {
	Int i,i1,i2,i3,i4,n=data.size();
	Doub c1=0.5,c2,h1r,h1i,h2r,h2i,wr,wi,wpr,wpi,wtemp;
	Doub theta=3.141592653589793238/Doub(n>>1);
	if (isign == 1) {
		c2 = -0.5;
		four1(data,1);
	} else {
		c2=0.5;
		theta = -theta;
	}
	wtemp=sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi=sin(theta);
	wr=1.0+wpr;
	wi=wpi;
	for (i=1;i<(n>>2);i++) {
		i2=1+(i1=i+i);
		i4=1+(i3=n-i1);
		h1r=c1*(data[i1]+data[i3]);
		h1i=c1*(data[i2]-data[i4]);
		h2r= -c2*(data[i2]+data[i4]);
		h2i=c2*(data[i1]-data[i3]);
		data[i1]=h1r+wr*h2r-wi*h2i;
		data[i2]=h1i+wr*h2i+wi*h2r;
		data[i3]=h1r-wr*h2r+wi*h2i;
		data[i4]= -h1i+wr*h2i+wi*h2r;
		wr=(wtemp=wr)*wpr-wi*wpi+wr;
		wi=wi*wpr+wtemp*wpi+wi;
	}
	if (isign == 1) {
		data[0] = (h1r=data[0])+data[1];
		data[1] = h1r-data[1];
	} else {
		data[0]=c1*((h1r=data[0])+data[1]);
		data[1]=c1*(h1r-data[1]);
		four1(data,-1);
	}
}
void sinft(VecDoub_IO &y) {
	Int j,n=y.size();
	Doub sum,y1,y2,theta,wi=0.0,wr=1.0,wpi,wpr,wtemp;
	theta=3.141592653589793238/Doub(n);
	wtemp=sin(0.5*theta);
	wpr= -2.0*wtemp*wtemp;
	wpi=sin(theta);
	y[0]=0.0;
	for (j=1;j<(n>>1)+1;j++) {
		wr=(wtemp=wr)*wpr-wi*wpi+wr;
		wi=wi*wpr+wtemp*wpi+wi;
		y1=wi*(y[j]+y[n-j]);
		y2=0.5*(y[j]-y[n-j]);
		y[j]=y1+y2;
		y[n-j]=y1-y2;
	}
	realft(y,1);
	y[0]*=0.5;
	sum=y[1]=0.0;
	for (j=0;j<n-1;j+=2) {
		sum += y[j];
		y[j]=y[j+1];
		y[j+1]=sum;
	}
}
void cosft1(VecDoub_IO &y) {
	const Doub PI=3.141592653589793238;
	Int j,n=y.size()-1;
	Doub sum,y1,y2,theta,wi=0.0,wpi,wpr,wr=1.0,wtemp;
	VecDoub yy(n);
	theta=PI/n;
	wtemp=sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi=sin(theta);
	sum=0.5*(y[0]-y[n]);
	yy[0]=0.5*(y[0]+y[n]);
	for (j=1;j<n/2;j++) {
		wr=(wtemp=wr)*wpr-wi*wpi+wr;
		wi=wi*wpr+wtemp*wpi+wi;
		y1=0.5*(y[j]+y[n-j]);
		y2=(y[j]-y[n-j]);
		yy[j]=y1-wi*y2;
		yy[n-j]=y1+wi*y2;
		sum += wr*y2;
	}
	yy[n/2]=y[n/2];
	realft(yy,1);
	for (j=0;j<n;j++) y[j]=yy[j];
	y[n]=y[1];
	y[1]=sum;
	for (j=3;j<n;j+=2) {
		sum += y[j];
		y[j]=sum;
	}
}
void cosft2(VecDoub_IO &y, const Int isign) {
	const Doub PI=3.141592653589793238;
	Int i,n=y.size();
	Doub sum,sum1,y1,y2,ytemp,theta,wi=0.0,wi1,wpi,wpr,wr=1.0,wr1,wtemp;
	theta=0.5*PI/n;
	wr1=cos(theta);
	wi1=sin(theta);
	wpr = -2.0*wi1*wi1;
	wpi=sin(2.0*theta);
	if (isign == 1) {
		for (i=0;i<n/2;i++) {
			y1=0.5*(y[i]+y[n-1-i]);
			y2=wi1*(y[i]-y[n-1-i]);
			y[i]=y1+y2;
			y[n-1-i]=y1-y2;
			wr1=(wtemp=wr1)*wpr-wi1*wpi+wr1;
			wi1=wi1*wpr+wtemp*wpi+wi1;
		}
		realft(y,1);
		for (i=2;i<n;i+=2) {
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
			y1=y[i]*wr-y[i+1]*wi;
			y2=y[i+1]*wr+y[i]*wi;
			y[i]=y1;
			y[i+1]=y2;
		}
		sum=0.5*y[1];
		for (i=n-1;i>0;i-=2) {
			sum1=sum;
			sum += y[i];
			y[i]=sum1;
		}
	} else if (isign == -1) {
		ytemp=y[n-1];
		for (i=n-1;i>2;i-=2)
			y[i]=y[i-2]-y[i];
		y[1]=2.0*ytemp;
		for (i=2;i<n;i+=2) {
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
			y1=y[i]*wr+y[i+1]*wi;
			y2=y[i+1]*wr-y[i]*wi;
			y[i]=y1;
			y[i+1]=y2;
		}
		realft(y,-1);
		for (i=0;i<n/2;i++) {
			y1=y[i]+y[n-1-i];
			y2=(0.5/wi1)*(y[i]-y[n-1-i]);
			y[i]=0.5*(y1+y2);
			y[n-1-i]=0.5*(y1-y2);
			wr1=(wtemp=wr1)*wpr-wi1*wpi+wr1;
			wi1=wi1*wpr+wtemp*wpi+wi1;
		}
	}
}
