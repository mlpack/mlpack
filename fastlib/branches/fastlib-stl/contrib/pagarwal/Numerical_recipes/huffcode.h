struct Huffcode {
	Int nch,nodemax,mq;
	Int ilong,nlong;
	VecInt ncod,left,right;
	VecUint icod;
	Uint setbit[32];

	Huffcode(const Int nnch, VecInt_I &nfreq)
	: nch(nnch), mq(2*nch-1), icod(mq), ncod(mq), left(mq), right(mq) {
		Int ibit,j,node,k,n,nused;
		VecInt index(mq), nprob(mq), up(mq);
		for (j=0;j<32;j++) setbit[j] = 1 << j;
		for (nused=0,j=0;j<nch;j++) {
			nprob[j]=nfreq[j];
			icod[j]=ncod[j]=0;
			if (nfreq[j] != 0) index[nused++]=j;
		}
		for (j=nused-1;j>=0;j--)
			heep(index,nprob,nused,j);
		k=nch;
		while (nused > 1) {
			node=index[0];
			index[0]=index[(nused--)-1];
			heep(index,nprob,nused,0);
			nprob[k]=nprob[index[0]]+nprob[node];
			left[k]=node;
			right[k++]=index[0];
			up[index[0]] = -Int(k);
			index[0]=k-1;
			up[node]=k;
			heep(index,nprob,nused,0);
		}
		up[(nodemax=k)-1]=0;
		for (j=0;j<nch;j++) {
			if (nprob[j] != 0) {
				for (n=0,ibit=0,node=up[j];node;node=up[node-1],ibit++) {
					if (node < 0) {
						n |= setbit[ibit];
						node = -node;
					}
				}
				icod[j]=n;
				ncod[j]=ibit;
			}
		}
		nlong=0;
		for (j=0;j<nch;j++) {
			if (ncod[j] > nlong) {
				nlong=ncod[j];
				ilong=j;
			}
		}
		if (nlong > numeric_limits<Uint>::digits)
			throw("Code too long in Huffcode.  See text.");
	}

	void codeone(const Int ich, char *code, Int &nb) {
		Int m,n,nc;
		if (ich >= nch) throw("bad ich (out of range) in Huffcode");
		if (ncod[ich]==0) throw("bad ich (zero prob) in Huffcode");
		for (n=ncod[ich]-1;n >= 0;n--,++nb) {
			nc=nb >> 3;
			m=nb & 7;
			if (m == 0) code[nc]=0;
			if ((icod[ich] & setbit[n]) != 0) code[nc] |= setbit[m];
		}
	}

	Int decodeone(char *code, Int &nb) {
		Int nc;
		Int node=nodemax-1;
		for (;;) {
			nc=nb >> 3;
			node=((code[nc] & setbit[7 & nb++]) != 0 ?
				right[node] : left[node]);
			if (node < nch) return node;
		}
	}

	void heep(VecInt_IO &index, VecInt_IO &nprob, const Int n, const Int m) {
		Int i=m,j,k;
		k=index[i];
		while (i < (n >> 1)) {
			if ((j = 2*i+1) < n-1
				&& nprob[index[j]] > nprob[index[j+1]]) j++;
			if (nprob[k] <= nprob[index[j]]) break;
			index[i]=index[j];
			i=j;
		}
		index[i]=k;
	}
};
