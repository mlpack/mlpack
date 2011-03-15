void fourn(Doub *data, VecInt_I &nn, const Int isign) {
	Int idim,i1,i2,i3,i2rev,i3rev,ip1,ip2,ip3,ifp1,ifp2;
	Int ibit,k1,k2,n,nprev,nrem,ntot=1,ndim=nn.size();
	Doub tempi,tempr,theta,wi,wpi,wpr,wr,wtemp;
	for (idim=0;idim<ndim;idim++) ntot *= nn[idim];
	if (ntot<2 || ntot&(ntot-1)) throw("must have powers of 2 in fourn");
	nprev=1;
	for (idim=ndim-1;idim>=0;idim--) {
		n=nn[idim];
		nrem=ntot/(n*nprev);
		ip1=nprev << 1;
		ip2=ip1*n;
		ip3=ip2*nrem;
		i2rev=0;
		for (i2=0;i2<ip2;i2+=ip1) {
			if (i2 < i2rev) {
				for (i1=i2;i1<i2+ip1-1;i1+=2) {
					for (i3=i1;i3<ip3;i3+=ip2) {
						i3rev=i2rev+i3-i2;
						SWAP(data[i3],data[i3rev]);
						SWAP(data[i3+1],data[i3rev+1]);
					}
				}
			}
			ibit=ip2 >> 1;
			while (ibit >= ip1 && i2rev+1 > ibit) {
				i2rev -= ibit;
				ibit >>= 1;
			}
			i2rev += ibit;
		}
		ifp1=ip1;
		while (ifp1 < ip2) {
			ifp2=ifp1 << 1;
			theta=isign*6.28318530717959/(ifp2/ip1);
			wtemp=sin(0.5*theta);
			wpr= -2.0*wtemp*wtemp;
			wpi=sin(theta);
			wr=1.0;
			wi=0.0;
			for (i3=0;i3<ifp1;i3+=ip1) {
				for (i1=i3;i1<i3+ip1-1;i1+=2) {
					for (i2=i1;i2<ip3;i2+=ifp2) {
						k1=i2;
						k2=k1+ifp1;
						tempr=wr*data[k2]-wi*data[k2+1];
						tempi=wr*data[k2+1]+wi*data[k2];
						data[k2]=data[k1]-tempr;
						data[k2+1]=data[k1+1]-tempi;
						data[k1] += tempr;
						data[k1+1] += tempi;
					}
				}
				wr=(wtemp=wr)*wpr-wi*wpi+wr;
				wi=wi*wpr+wtemp*wpi+wi;
			}
			ifp1=ifp2;
		}
		nprev *= n;
	}
}

void fourn(VecDoub_IO &data, VecInt_I &nn, const Int isign) {
	fourn(&data[0],nn,isign);
}
void rlft3(Doub *data, Doub *speq, const Int isign,
	const Int nn1, const Int nn2, const Int nn3) {
	Int i1,i2,i3,j1,j2,j3,k1,k2,k3,k4;
	Doub theta,wi,wpi,wpr,wr,wtemp;
	Doub c1,c2,h1r,h1i,h2r,h2i;
	VecInt nn(3);
	VecDoubp spq(nn1);
	for (i1=0;i1<nn1;i1++) spq[i1] = speq + 2*nn2*i1;
	c1 = 0.5;
	c2 = -0.5*isign;
	theta = isign*(6.28318530717959/nn3);
	wtemp = sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi = sin(theta);
	nn[0] = nn1;
	nn[1] = nn2;
	nn[2] = nn3 >> 1;
	if (isign == 1) {
		fourn(data,nn,isign);
		k1=0;
		for (i1=0;i1<nn1;i1++)
			for (i2=0,j2=0;i2<nn2;i2++,k1+=nn3) {
				spq[i1][j2++]=data[k1];
				spq[i1][j2++]=data[k1+1];
			}
	}
	for (i1=0;i1<nn1;i1++) {
		j1=(i1 != 0 ? nn1-i1 : 0);
		wr=1.0;
		wi=0.0;
		for (i3=0;i3<=(nn3>>1);i3+=2) {
			k1=i1*nn2*nn3;
			k3=j1*nn2*nn3;
			for (i2=0;i2<nn2;i2++,k1+=nn3) {
				if (i3 == 0) {
					j2=(i2 != 0 ? ((nn2-i2)<<1) : 0);
					h1r=c1*(data[k1]+spq[j1][j2]);
					h1i=c1*(data[k1+1]-spq[j1][j2+1]);
					h2i=c2*(data[k1]-spq[j1][j2]);
					h2r= -c2*(data[k1+1]+spq[j1][j2+1]);
					data[k1]=h1r+h2r;
					data[k1+1]=h1i+h2i;
					spq[j1][j2]=h1r-h2r;
					spq[j1][j2+1]=h2i-h1i;
				} else {
					j2=(i2 != 0 ? nn2-i2 : 0);
					j3=nn3-i3;
					k2=k1+i3;
					k4=k3+j2*nn3+j3;
					h1r=c1*(data[k2]+data[k4]);
					h1i=c1*(data[k2+1]-data[k4+1]);
					h2i=c2*(data[k2]-data[k4]);
					h2r= -c2*(data[k2+1]+data[k4+1]);
					data[k2]=h1r+wr*h2r-wi*h2i;
					data[k2+1]=h1i+wr*h2i+wi*h2r;
					data[k4]=h1r-wr*h2r+wi*h2i;
					data[k4+1]= -h1i+wr*h2i+wi*h2r;
				}
			}
			wr=(wtemp=wr)*wpr-wi*wpi+wr;
			wi=wi*wpr+wtemp*wpi+wi;
		}
	}
	if (isign == -1) fourn(data,nn,isign);
}

void rlft3(Mat3DDoub_IO &data, MatDoub_IO &speq, const Int isign) {
	if (speq.nrows() != data.dim1() || speq.ncols() != 2*data.dim2())
		throw("bad dims in rlft3");
	rlft3(&data[0][0][0],&speq[0][0],isign,data.dim1(),data.dim2(),data.dim3());
}

void rlft3(MatDoub_IO &data, VecDoub_IO &speq, const Int isign) {
	if (speq.size() != 2*data.nrows()) throw("bad dims in rlft3");
	rlft3(&data[0][0],&speq[0],isign,1,data.nrows(),data.ncols());
}
