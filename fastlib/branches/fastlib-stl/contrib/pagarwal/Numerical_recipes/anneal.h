struct Anneal {
	Ranq1 myran;
	Anneal() : myran(1234) {}
	void order(VecDoub_I &x, VecDoub_I &y, VecInt_IO &iorder)
	{
		const Doub TFACTR=0.9;
		Bool ans;
		Int i,i1,i2,nn;
		VecInt n(6);
		Doub de,path=0.0,t=0.5;
		Int ncity=x.size();
		Int nover=100*ncity;
		Int nlimit=10*ncity;
		for (i=0;i<ncity-1;i++) {
			i1=iorder[i];
			i2=iorder[i+1];
			path += alen(x[i1],x[i2],y[i1],y[i2]);
		}
		i1=iorder[ncity-1];
		i2=iorder[0];
		path += alen(x[i1],x[i2],y[i1],y[i2]);
		cout << fixed << setprecision(6);
		for (Int j=0;j<100;j++) {
			Int nsucc=0;
			for (Int k=0;k<nover;k++) {
				do {
					n[0]=Int(ncity*myran.doub());
					n[1]=Int((ncity-1)*myran.doub());
					if (n[1] >= n[0]) ++n[1];
					nn=(n[0]-n[1]+ncity-1) % ncity;
				} while (nn<2);
				if (myran.doub() < 0.5) {
					n[2]=n[1]+Int(abs(nn-1)*myran.doub())+1;
					n[2] %= ncity;
					de=trncst(x,y,iorder,n);
					ans=metrop(de,t);
					if (ans) {
						++nsucc;
						path += de;
						trnspt(iorder,n);
					}
				} else {
					de=revcst(x,y,iorder,n);
					ans=metrop(de,t);
					if (ans) {
						++nsucc;
						path += de;
						reverse(iorder,n);
					}
				}
				if (nsucc >= nlimit) break;
			}
			cout << endl << "T = " << setw(12) << t;
			cout << "	 Path Length = " << setw(12) << path << endl;
			cout << "Successful Moves: " << nsucc << endl;
			t *= TFACTR;
			if (nsucc == 0) return;
		}
	}
	
	Doub revcst(VecDoub_I &x, VecDoub_I &y, VecInt_I &iorder, VecInt_IO &n)
	{
		VecDoub xx(4),yy(4);
		Int ncity=x.size();
		n[2]=(n[0]+ncity-1) % ncity;
		n[3]=(n[1]+1) % ncity;
		for (Int j=0;j<4;j++) {
			Int ii=iorder[n[j]];
			xx[j]=x[ii];
			yy[j]=y[ii];
		}
		Doub de = -alen(xx[0],xx[2],yy[0],yy[2]);
		de -= alen(xx[1],xx[3],yy[1],yy[3]);
		de += alen(xx[0],xx[3],yy[0],yy[3]);
		de += alen(xx[1],xx[2],yy[1],yy[2]);
		return de;
	}
	
	void reverse(VecInt_IO &iorder, VecInt_I &n)
	{
		Int ncity=iorder.size();
		Int nn=(1+((n[1]-n[0]+ncity) % ncity))/2;
		for (Int j=0;j<nn;j++) {
			Int k=(n[0]+j) % ncity;
			Int l=(n[1]-j+ncity) % ncity;
			Int itmp=iorder[k];
			iorder[k]=iorder[l];
			iorder[l]=itmp;
		}
	}
	
	Doub trncst(VecDoub_I &x, VecDoub_I &y, VecInt_I &iorder, VecInt_IO &n)
	{
		VecDoub xx(6),yy(6);
		Int ncity=x.size();
		n[3]=(n[2]+1) % ncity;
		n[4]=(n[0]+ncity-1) % ncity;
		n[5]=(n[1]+1) % ncity;
		for (Int j=0;j<6;j++) {
			Int ii=iorder[n[j]];
			xx[j]=x[ii];
			yy[j]=y[ii];
		}
		Doub de = -alen(xx[1],xx[5],yy[1],yy[5]);
		de -= alen(xx[0],xx[4],yy[0],yy[4]);
		de -= alen(xx[2],xx[3],yy[2],yy[3]);
		de += alen(xx[0],xx[2],yy[0],yy[2]);
		de += alen(xx[1],xx[3],yy[1],yy[3]);
		de += alen(xx[4],xx[5],yy[4],yy[5]);
		return de;
	}
	
	void trnspt(VecInt_IO &iorder, VecInt_I &n)
	{
		Int ncity=iorder.size();
		VecInt jorder(ncity);
		Int m1=(n[1]-n[0]+ncity) % ncity;
		Int m2=(n[4]-n[3]+ncity) % ncity;
		Int m3=(n[2]-n[5]+ncity) % ncity;
		Int nn=0;
		for (Int j=0;j<=m1;j++) {
			Int jj=(j+n[0]) % ncity;
			jorder[nn++]=iorder[jj];
		}
		for (Int j=0;j<=m2;j++) {
			Int jj=(j+n[3]) % ncity;
			jorder[nn++]=iorder[jj];
		}
		for (Int j=0;j<=m3;j++) {
			Int jj=(j+n[5]) % ncity;
			jorder[nn++]=iorder[jj];
		}
		for (Int j=0;j<ncity;j++)
			iorder[j]=jorder[j];
	}
	
	Bool metrop(const Doub de, const Doub t)
	{
		return de < 0.0 || myran.doub() < exp(-de/t);
	}

	inline Doub alen(const Doub a, const Doub b, const Doub c, const Doub d)
	{
		return sqrt((b-a)*(b-a)+(d-c)*(d-c));
	}
};
