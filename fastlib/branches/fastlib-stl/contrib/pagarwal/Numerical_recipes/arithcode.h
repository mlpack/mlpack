struct Arithcode {
	Int nch,nrad,ncum;
	Uint jdif,nc,minint;
	VecUint ilob,iupb;
	VecInt ncumfq;
	static const Int NWK=20;

	Arithcode(VecInt_I &nfreq, const Int nnch, const Int nnrad)
	: nch(nnch), nrad(nnrad), ilob(NWK), iupb(NWK), ncumfq(nch+2) {
		Int j;
		if (nrad > 256) throw("output radix must be <= 256 in Arithcode");
		minint=numeric_limits<Uint>::max()/nrad;
		ncumfq[0]=0;
		for (j=1;j<=nch;j++) ncumfq[j]=ncumfq[j-1]+MAX(nfreq[j-1],1);
		ncum=ncumfq[nch+1]=ncumfq[nch]+1;
	}

	void messageinit() {
		Int j;
		jdif=nrad-1;
		for (j=NWK-1;j>=0;j--) {
			iupb[j]=nrad-1;
			ilob[j]=0;
			nc=j;
			if (jdif > minint) return;
			jdif=(jdif+1)*nrad-1;
		}
		throw("NWK too small in arcode.");
	}

	void codeone(const Int ich, char *code, Int &lcd) {
		if (ich > nch) throw("bad ich in Arithcode");
		advance(ich,code,lcd,1);
	}

	Int decodeone(char *code, Int &lcd) {
		Int ich;
		Uint j,ihi,ja,m;
		ja=(Uchar) code[lcd]-ilob[nc];
		for (j=nc+1;j<NWK;j++) {
			ja *= nrad;
			ja += Uchar(code[lcd+j-nc])-ilob[j];
		}
		ihi=nch+1;
		ich=0;
		while (ihi-ich > 1) {
			m=(ich+ihi)>>1;
			if (ja >= multdiv(jdif,ncumfq[m],ncum)) ich=m;
			else ihi=m;
		}
		if (ich != nch) advance(ich,code,lcd,-1);
		return ich;
	}

	void advance(const Int ich, char *code, Int &lcd, const Int isign) {
		Uint j,k,jh,jl;
		jh=multdiv(jdif,ncumfq[ich+1],ncum);
		jl=multdiv(jdif,ncumfq[ich],ncum);
		jdif=jh-jl;
		arrsum(ilob,iupb,jh,NWK,nrad,nc);
		arrsum(ilob,ilob,jl,NWK,nrad,nc);
		for (j=nc;j<NWK;j++) {
			if (ich != nch && iupb[j] != ilob[j]) break;
			if (isign > 0) code[lcd] = ilob[j];
			lcd++;
		}
		if (j+1 > NWK) return;
		nc=j;
		for(j=0;jdif<minint;j++)
			jdif *= nrad;
		if (j > nc) throw("NWK too small in arcode.");
		if (j != 0) {
			for (k=nc;k<NWK;k++) {
				iupb[k-j]=iupb[k];
				ilob[k-j]=ilob[k];
			}
		}
		nc -= j;
		for (k=NWK-j;k<NWK;k++) iupb[k]=ilob[k]=0;
		return;
	}

	inline Uint multdiv(const Uint j, const Uint k, const Uint m) {
		return Uint((Ullong(j)*Ullong(k)/Ullong(m)));
	}

	void arrsum(VecUint_I &iin, VecUint_O &iout, Uint ja,
	const Int nwk, const Uint nrad, const Uint nc) {
		Uint karry=0,j,jtmp;
		for (j=nwk-1;j>nc;j--) {
			jtmp=ja;
			ja /= nrad;
			iout[j]=iin[j]+(jtmp-ja*nrad)+karry;
			if (iout[j] >= nrad) {
				iout[j] -= nrad;
				karry=1;
			} else karry=0;
		}
		iout[nc]=iin[nc]+ja+karry;
	}
};
