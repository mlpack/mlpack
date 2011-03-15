struct Icrc {

	Uint jcrc,jfill,poly;
	static Uint icrctb[256];

	Icrc(const Int jpoly, const Bool fill=true) : jfill(fill ? 255 : 0) {
		Int j;
		Uint okpolys[8] = {0x755B,0xA7D3,0x8005,0x1021,0x5935,0x90D9,0x5B93,0x2D17};
		poly = okpolys[jpoly & 7];
		for (j=0;j<256;j++) {
			icrctb[j]=icrc1(j << 8,0);
		}
		jcrc = (jfill | (jfill << 8));
	}

	Uint crc(const string &bufptr) {
		jcrc = (jfill | (jfill << 8));
		return concat(bufptr);
	}

	Uint concat(const string &bufptr) {
		Uint j,cword=jcrc,len=bufptr.size();
		for (j=0;j<len;j++) {
			cword=icrctb[Uchar(bufptr[j]) ^ hibyte(cword)] ^ (lobyte(cword) << 8);
		}
		return jcrc = cword;
	}

	Uint icrc1(const Uint jcrc, const Uchar onech) {
		Int i;
		Uint ans=(jcrc ^ onech << 8);
		for (i=0;i<8;i++) {
			if (ans & 0x8000) ans = (ans <<= 1) ^ poly;
			else ans <<= 1;
			ans &= 0xffff;
		}
		return ans;
	}

	inline Uchar lobyte(const unsigned short x) {
		return (Uchar)(x & 0xff); }
	inline Uchar hibyte(const unsigned short x) {
		return (Uchar)((x >> 8) & 0xff); }
};
Uint Icrc::icrctb[256];
