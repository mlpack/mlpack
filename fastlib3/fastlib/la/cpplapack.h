#ifndef CPP_LAPACK_H_
#define CPP_LAPACK_H_
#include "clapack.h"
/**
 * @file cpplapack.h
 *
 * @brief A template version of LAPACK
 */

namespace la{

template<typename Precision>
CppLapack {
};

template<>
CppLapack<float> {
 public:
  CppLapack() {
    Precision fake_matrix[64];
    Precision fake_workspace;
    Precision fake_vector;
    f77_integer fake_pivots;
    f77_integer fake_info;
      
    /* TODO: This may want to be ilaenv */
    this->getri(1, fake_matrix, 1, &fake_pivots, &fake_workspace,
        -1, &fake_info);
    this->getri_block_size = int(fake_workspace);
      
    this->geqrf(1, 1, fake_matrix, 1, &fake_vector, &fake_workspace, -1,
        &fake_info);
    this->geqrf_block_size = int(fake_workspace);
      
    this->orgqr(1, 1, 1, fake_matrix, 1, &fake_vector, &fake_workspace, -1,
         &fake_info);
    this->orgqr_block_size = int(fake_workspace);
      
    this->geqrf_dorgqr_block_size =
         std::max(la::dgeqrf_block_size, la::dorgqr_block_size);
 
  }
  static int getri_block_size;
  static int geqrf_block_size;
  static int orgqr_block_size;
  static int geqrf_dorgqr_block_size;


  static inline void bdsdc(const char *uplo, const char *compq, index_t &n, float *d__, float *e, float *u, index_t &ldu, float *vt, index_t &ldvt, float *q, index_t *iq, float *work, index_t *iwork, index_t *info) {
   inline F77_FUNC(bdsdc)(uplo, compq, n, d__, e, u, ldu, vt, ldvt, q, iq, work, iwork, info); 
  }
  static inline void bdsqr(const char *uplo, index_t &n, index_t &ncvt, index_t &nru, index_t &ncc, float *d__, float *e, float *vt, index_t &ldvt, float *u, index_t &ldu, float *c__, index_t &ldc, float *work, index_t *info) {
    inline F77_FUNC(bdsqr)(uplo, n, ncvt, nru, ncc, d__, e, vt, ldvt, u, ldu, c__, ldc, work, info);
  }
  static inline void disna(const char *job, index_t &m, index_t &n, float *d__, float *sep, index_t *info) {
    inline F77_FUNC(disna)(job, m, n, d__, sep, info);
  }
  static inline void gbbrd(const char *vect, index_t &m, index_t &n, index_t &ncc, index_t *kl, index_t *ku, float *ab, index_t &ldab, float *d__, float *e, float *q, index_t &ldq, float *pt, index_t &ldpt, float *c__, index_t &ldc, float *work, index_t *info) {
    inline F77_FUNC(gbbrd)(vect, m, n, ncc, kl, ku, ab, ldab, d__, e, q, ldq, pt, ldpt, c__, ldc, work, info);
  }
  static inline void gbcon(const char *norm, index_t &n, index_t *kl, index_t *ku, float *ab, index_t &ldab, index_t *ipiv, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info) {
    inline F77_FUNC(gbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork, info);
  }
  static inline void gbequ(index_t &m, index_t &n, index_t *kl, index_t *ku, float *ab, index_t &ldab, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, index_t *info) {
    inline F77_FUNC(gbequ)(m, n, kl, ku, ab, ldab, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gbrfs(const char *trans, index_t &n, index_t *kl, index_t *ku, index_t &nrhs, float *ab, index_t &ldab, float *afb, index_t &ldafb, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t*iwork, index_t *info) {
    inline F77_FUNC(gbrfs)(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, *x, ldx, ferr, berr, work, iwork, info);
  }
  static inline void gbsv(index_t &n, index_t *kl, index_t *ku, index_t &nrhs, float *ab, index_t &ldab, index_t *ipiv, float *b, index_t &ldb, index_t *info) {
    inline F77_FUNC(gbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
  }
  static inline void gbsvx(const char *fact, const char *trans, index_t &n, index_t *kl, index_t *ku, index_t &nrhs, float *ab, index_t &ldab, float *afb, index_t &ldafb, index_t *ipiv, const char *equed, float *r__, float *c__, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info) {
    inline F77_FUNC(gbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r__, c__, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
  }
  static inline void gbtf2(index_t &m, index_t &n, index_t *kl, index_t *ku, float *ab, index_t &ldab, index_t *ipiv, index_t *info) {
    inline F77_FUNC(gbtf2)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrf(index_t &m, index_t &n, index_t *kl, index_t *ku, float *ab, index_t &ldab, index_t *ipiv, index_t *info) {
    inline F77_FUNC(gbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrs(const char *trans, index_t &n, index_t *kl, index_t *ku, index_t &nrhs, float *ab, index_t &ldab, index_t *ipiv, float *b, index_t &ldb, index_t *info) {
    inline F77_FUNC(gbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);

  }
  static inline void gebak(const char *job, const char *side, index_t &n, index_t *ilo, index_t *ihi, float *scale, index_t &m, float *v, index_t &ldv, index_t *info) {
    inline F77_FUNC(gebak)(job, side, n, ilo, ihi, scale, m, v, ldv, info);
  }
  static inline void gebal(const char *job, index_t &n, float *a, index_t &lda, index_t *ilo, index_t *ihi, float *scale, index_t *info) {
    inline F77_FUNC(gebal)(job, n, a, lda, ilo, ihi, scale, info);

  }
  static inline void gebd2(index_t &m, index_t &n, float *a, index_t &lda, float *d__, float *e, float *tauq, float *taup, float *work, index_t *info) {
    inline F77_FUNC(gebd2)(m, n, a, lda, d__, e, tauq, taup, work, info);
  }
  static inline void gebrd(index_t &m, index_t &n, float *a, index_t &lda, float *d__, float *e, float *tauq, float *taup, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(gebrd)(m, n, a, lda, d__, e, tauq, taup, work, lwork, info);
  }
  static inline void gecon(const char *norm, index_t &n, float *a, index_t &lda, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info) {
    inline F77_FUNC(gecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
  }
  static inline void geequ(index_t &m, index_t &n, float *a, index_t &lda, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, index_t *info) {
    inline F77_FUNC(geequ)(m, n, a, lda, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gees(const char *jobvs, const char *sort, f77_logical_func select, index_t &n, float *a, index_t &lda, index_t *sdim, float *wr, float *wi, float *vs, index_t &ldvs, float *work, index_t &lwork, unsigned int *bwork, index_t *info) {
    inline F77_FUNC(gees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
  }
  static inline void geesx(const char *jobvs, const char *sort, f77_logical_func select, const char *sense, index_t &n, float *a, index_t &lda, index_t *sdim, float *wr, float *wi, float *vs, index_t &ldvs, float *rconde, float *rcondv, float *work, index_t &lwork, index_t *iwork, index_t &liwork, unsigned int *bwork, index_t *info) {
    inline F77_FUNC(geesx)(jobvs, sort, select, sense, n, a, lda, sdim, wr, wi, vs, ldvs, rconde, rcondv, work, lwork, iwork, liwork, bwork, info);
  }
  static inline void geev(const char *jobvl, const char *jobvr, index_t &n, float *a, index_t &lda, float *wr, float *wi, float *vl, index_t &ldvl, float *vr, index_t &ldvr, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(geev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
  }
  static inline void geevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, index_t &n, float *a, index_t &lda, float *wr, float *wi, float *vl, index_t &ldvl, float *vr, index_t &ldvr, index_t *ilo, index_t *ihi, float *scale, float *abnrm, float *rconde, float *rcondv, float *work, index_t &lwork, index_t *iwork, index_t *info) {
    inline F77_FUNC(geevx()balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work,lwork, iwork, info);
  }
  static inline void gegs(const char *jobvsl, const char *jobvsr, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *vsl, index_t &ldvsl, float *vsr, index_t &ldvsr, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(gegs)(jobvsl, jobvsr, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, info);
  }
  static inline void gegv(const char *jobvl, const char *jobvr, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *vl, index_t &ldvl, float *vr, index_t &ldvr, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(gegv)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, index_t &ldvl, vr, ldvr, work, lwork, info);
  }
//  static inline void gehd2(index_t &n, index_t *ilo, index_t *ihi, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void gehrd(index_t &n, index_t *ilo, index_t *ihi, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void gelq2(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void gelqf(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void gels(const char *trans, index_t &m, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, float *work, index_t &lwork, index_t *info);
//  static inline void gelsd(index_t &m, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, float *s, float *rcond, index_t *rank, float *work, index_t &lwork, index_t *iwork, index_t *info);
//  static inline void gelss(index_t &m, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, float *s, float *rcond, index_t *rank, float *work, index_t &lwork, index_t *info);
//  static inline void gelsx(index_t &m, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, index_t *jpvt, float *rcond, index_t *rank, float *work, index_t *info);
//  static inline void gelsy(index_t &m, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, index_t *jpvt, float *rcond, index_t *rank, float *work, index_t &lwork, index_t *info);
//  static inline void geql2(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void geqlf(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
// static inline void geqp3(index_t &m, index_t &n, float *a, index_t &lda, index_t *jpvt, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void geqpf(index_t &m, index_t &n, float *a, index_t &lda, index_t *jpvt, float *tau, float *work, index_t *info);
//  static inline void geqr2(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t *info);
  static inline void geqrf(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(geqrf)(m, n, a, lda, tau, work, lwork, info);
  }
  static inline void gerfs(const char *trans, index_t &n, index_t &nrhs, float *a, index_t &lda, float *af, index_t &ldaf, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info) {
    inline F77_FUNC(gerfs)(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  
  }
//  static inline void gerq2(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void gerqf(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void gesc2(index_t &n, float *a, index_t &lda, float *rhs, index_t *ipiv, index_t *jpiv, float *scale);
  static inline void gesdd(const char *jobz, index_t &m, index_t &n, float *a, index_t &lda, float *s, float *u, index_t &ldu, float *vt, index_t &ldvt, float *work, index_t &lwork, index_t *iwork, index_t *info) {
    inline F77_FUNC(gesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static inline void gesv(index_t &n, index_t &nrhs, float *a, index_t &lda, index_t *ipiv, float *b, index_t &ldb, index_t *info) {
    inline F77_FUNC(gesv)(index_t &n, index_t &nrhs, float *a, index_t &lda, index_t *ipiv, float *b, index_t &ldb, index_t *info);
  }
//  static inline void gesvd(const char *jobu, const char *jobvt, index_t &m, index_t &n, float *a, index_t &lda, float *s, float *u, index_t &ldu, float *vt, index_t &ldvt, float *work, index_t &lwork, index_t *info);
//  static inline void gesvx(const char *fact, const char *trans, index_t &n, index_t &nrhs, float *a, index_t &lda, float *af, index_t &ldaf, index_t *ipiv, const char *equed, float *r__, float *c__, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void getc2(index_t &n, float *a, index_t &lda, index_t *ipiv, index_t *jpiv, index_t *info);
//  static inline void getf2(index_t &m, index_t &n, float *a, index_t &lda, index_t *ipiv, index_t *info);
//  static inline void getrf(index_t &m, index_t &n, float *a, index_t &lda, index_t *ipiv, index_t *info);
//  static inline void getri(index_t &n, float *a, index_t &lda, index_t *ipiv, float *work, index_t &lwork, index_t *info);
//  static inline void getrs(const char *trans, index_t &n, index_t &nrhs, float *a, index_t &lda, index_t *ipiv, float *b, index_t &ldb, index_t *info);
//  static inline void ggbak(const char *job, const char *side, index_t &n, index_t *ilo, index_t *ihi, float *lscale, float *rscale, index_t &m, float *v, index_t &ldv, index_t *info);
//  static inline void ggbal(const char *job, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, index_t *ilo, index_t *ihi, float *lscale, float *rscale, float *work, index_t *info);
//  static inline void gges(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, index_t *sdim, float *alphar, float *alphai, float *beta, float *vsl, index_t &ldvsl, float *vsr, index_t &ldvsr, float *work, index_t &lwork, f77_logical *bwork, index_t *info);
//  static inline void ggesx(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, const char *sense, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, index_t *sdim, float *alphar, float *alphai, float *beta, float *vsl, index_t &ldvsl, float *vsr, index_t &ldvsr, float *rconde, float *rcondv, float *work, index_t &lwork, index_t *iwork, index_t &liwork, f77_logical *bwork, index_t *info);
//  static inline void ggev(const char *jobvl, const char *jobvr, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *vl, index_t &ldvl, float *vr, index_t &ldvr, float *work, index_t &lwork, index_t *info);
//  static inline void ggevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *vl, index_t &ldvl, float *vr, index_t &ldvr, index_t *ilo, index_t *ihi, float *lscale, float *rscale, float *abnrm, float *bbnrm, float *rconde, float *rcondv, float *work, index_t &lwork, index_t *iwork, f77_logical *bwork, index_t *info);
//  static inline void ggglm(index_t &n, index_t &m, index_t &p, float *a, index_t &lda, float *b, index_t &ldb, float *d__, float *x, float *y, float *work, index_t &lwork, index_t *info);
//  static inline void gghrd(const char *compq, const char *compz, index_t &n, index_t *ilo, index_t *ihi, float *a, index_t &lda, float *b, index_t &ldb, float *q, index_t &ldq, float *z__, index_t &ldz, index_t *info);
//  static inline void gglse(index_t &m, index_t &n, index_t &p, float *a, index_t &lda, float *b, index_t &ldb, float *c__, float *d__, float *x, float *work, index_t &lwork, index_t *info);
//  static inline void ggqrf(index_t &n, index_t &m, index_t &p, float *a, index_t &lda, float *taua, float *b, index_t &ldb, float *taub, float *work, index_t &lwork, index_t *info);
//  static inline void ggrqf(index_t &m, index_t &p, index_t &n, float *a, index_t &lda, float *taua, float *b, index_t &ldb, float *taub, float *work, index_t &lwork, index_t *info);
//  static inline void ggsvd(const char *jobu, const char *jobv, const char *jobq, index_t &m, index_t &n, index_t &p, index_t &k, index_t *l, float *a, index_t &lda, float *b, index_t &ldb, float *alpha, float *beta, float *u, index_t &ldu, float *v, index_t &ldv, float *q, index_t &ldq, float *work, index_t *iwork, index_t *info);
//  static inline void ggsvp(const char *jobu, const char *jobv, const char *jobq, index_t &m, index_t &p, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *tola, float *tolb, index_t &k, index_t *l, float *u, index_t &ldu, float *v, index_t &ldv, float *q, index_t &ldq, index_t *iwork, float *tau, float *work, index_t *info);
//  static inline void gtcon(const char *norm, index_t &n, float *dl, float *d__, float *du, float *du2, index_t *ipiv, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void gtrfs(const char *trans, index_t &n, index_t &nrhs, float *dl, float *d__, float *du, float *dlf, float *df, float *duf, float *du2, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void gtsv(index_t &n, index_t &nrhs, float *dl, float *d__, float *du, float *b, index_t &ldb, index_t *info);
//  static inline void gtsvx(const char *fact, const char *trans, index_t &n, index_t &nrhs, float *dl, float *d__, float *du, float *dlf, float *df, float *duf, float *du2, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void gttrf(index_t &n, float *dl, float *d__, float *du, float *du2, index_t *ipiv, index_t *info);
//  static inline void gttrs(const char *trans, index_t &n, index_t &nrhs, float *dl, float *d__, float *du, float *du2, index_t *ipiv, float *b, index_t &ldb, index_t *info);
//  static inline void gtts2(index_t *itrans, index_t &n, index_t &nrhs, float *dl, float *d__, float *du, float *du2, index_t *ipiv, float *b, index_t &ldb);
//  static inline void hgeqz(const char *job, const char *compq, const char *compz, index_t &n, index_t *ilo, index_t *ihi, float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *q, index_t &ldq, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *info);
//  static inline void hsein(const char *side, const char *eigsrc, const char *initv, f77_logical *select, index_t &n, float *h__, index_t &ldh, float *wr, float *wi, float *vl, index_t &ldvl, float *vr, index_t &ldvr, index_t *mm, index_t *m_out, float *work, index_t *ifaill, index_t *ifailr, index_t *info);
//  static inline void hseqr(const char *job, const char *compz, index_t &n, index_t *ilo, index_t *ihi, float *h__, index_t &ldh, float *wr, float *wi, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *info);
//  static inline void labad(float *small, float *large);
//  static inline void labrd(index_t &m, index_t &n, index_t *nb, float *a, index_t &lda, float *d__, float *e, float *tauq, float *taup, float *x, index_t &ldx, float *y, index_t &ldy);
//  static inline void lacon(index_t &n, float *v, float *x, index_t *isgn, float *est, index_t *kase);
//  static inline void lacpy(const char *uplo, index_t &m, index_t &n, float *a, index_t &lda, float *b, index_t &ldb);
//  static inline void ladiv(float *a, float *b, float *c__, float *d__, float *p, float *q);
//  static inline void lae2(float *a, float *b, float *c__, float *rt1, float *rt2);
//  static inline void laebz(index_t *ijob, index_t *nitmax, index_t &n, index_t *mmax, index_t *minp, index_t *nbmin, float *abstol, float *reltol, float *pivmin, float *d__, float *e, float *e2, index_t *nval, float *ab, float *c__, index_t *mout, index_t *nab, float *work, index_t *iwork, index_t *info);
//  static inline void laed0(index_t *icompq, index_t *qsiz, index_t &n, float *d__, float *e, float *q, index_t &ldq, float *qstore, index_t &ldqs, float *work, index_t *iwork, index_t *info);
//  static inline void laed1(index_t &n, float *d__, float *q, index_t &ldq, index_t *indxq, float *rho, index_t *cutpnt, float *work, index_t *iwork, index_t *info);
//  static inline void laed2(index_t &k, index_t &n, index_t *n1, float *d__, float *q, index_t &ldq, index_t *indxq, float *rho, float *z__, float *dlamda, float *w, float *q2, index_t *indx, index_t *indxc, index_t *indxp, index_t *coltyp, index_t *info);
//  static inline void laed3(index_t &k, index_t &n, index_t *n1, float *d__, float *q, index_t &ldq, float *rho, float *dlamda, float *q2, index_t *indx, index_t *ctot, float *w, float *s, index_t *info);
//  static inline void laed4(index_t &n, index_t *i__, float *d__, float *z__, float *delta, float *rho, float *dlam, index_t *info);
//  static inline void laed5(index_t *i__, float *d__, float *z__, float *delta, float *rho, float *dlam);
//  static inline void laed6(index_t *kniter, f77_logical *orgati, float *rho, float *d__, float *z__, float *finit, float *tau, index_t *info);
//  static inline void laed7(index_t *icompq, index_t &n, index_t *qsiz, index_t *tlvls, index_t *curlvl, index_t *curpbm, float *d__, float *q, index_t &ldq, index_t *indxq, float *rho, index_t *cutpnt, float *qstore, index_t *qptr, index_t *prmptr, index_t *perm, index_t *givptr, index_t *givcol, float *givnum, float *work, index_t *iwork, index_t *info);
//  static inline void laed8(index_t *icompq, index_t &k, index_t &n, index_t *qsiz, float *d__, float *q, index_t &ldq, index_t *indxq, float *rho, index_t *cutpnt, float *z__, float *dlamda, float *q2, index_t &ldq2, float *w, index_t *perm, index_t *givptr, index_t *givcol, float *givnum, index_t *indxp, index_t *indx, index_t *info);
//  static inline void laed9(index_t &k, index_t *kstart, index_t *kstop, index_t &n, float *d__, float *q, index_t &ldq, float *rho, float *dlamda, float *w, float *s, index_t &lds, index_t *info);
//  static inline void laeda(index_t &n, index_t *tlvls, index_t *curlvl, index_t *curpbm, index_t *prmptr, index_t *perm, index_t *givptr, index_t *givcol, float *givnum, float *q, index_t *qptr, float *z__, float *ztemp, index_t *info);
//  static inline void laein(f77_logical *rightv, f77_logical *noinit, index_t &n, float *h__, index_t &ldh, float *wr, float *wi, float *vr, float *vi, float *b, index_t &ldb, float *work, float *eps3, float *smlnum, float *bignum, index_t *info);
//  static inline void laev2(float *a, float *b, float *c__, float *rt1, float *rt2, float *cs1, float *sn1);
//  static inline void laexc(f77_logical &wantq, index_t &n, float *t, index_t &ldt, float *q, index_t &ldq, index_t *j1, index_t *n1, index_t *n2, float *work, index_t *info);
//  static inline void lag2(float *a, index_t &lda, float *b, index_t &ldb, float *safmin, float *scale1, float *scale2, float *wr1, float *wr2, float *wi);
//  static inline void lags2(f77_logical *upper, float *a1, float *a2, float *a3, float *b1, float *b2, float *b3, float *csu, float *snu, float *csv, float *snv, float *csq, float *snq);
//  static inline void lagtf(index_t &n, float *a, float *lambda, float *b, float *c__, float *tol, float *d__, index_t *in, index_t *info);
//  static inline void lagtm(const char *trans, index_t &n, index_t &nrhs, float *alpha, float *dl, float *d__, float *du, float *x, index_t &ldx, float *beta, float *b, index_t &ldb);
//  static inline void lagts(index_t *job, index_t &n, float *a, float *b, float *c__, float *d__, index_t *in, float *y, float *tol, index_t *info);
//  static inline void lagv2(float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *csl, float *snl, float *csr, float *snr);
//  static inline void lahqr(f77_logical &wantt, f77_logical &wantz, index_t &n, index_t *ilo, index_t *ihi, float *h__, index_t &ldh, float *wr, float *wi, index_t *iloz, index_t *ihiz, float *z__, index_t &ldz, index_t *info);
//  static inline void lahrd(index_t &n, index_t &k, index_t *nb, float *a, index_t &lda, float *tau, float *t, index_t &ldt, float *y, index_t &ldy);
//  static inline void laic1(index_t *job, index_t *j, float *x, float *sest, float *w, float *gamma, float *sestpr, float *s, float *c__);
//  static inline void laln2(f77_logical *ltrans, index_t *na, index_t *nw, float *smin, float *ca, float *a, index_t &lda, float *d1, float *d2, float *b, index_t &ldb, float *wr, float *wi, float *x, index_t &ldx, float *scale, float *xnorm, index_t *info);
//  static inline void lals0(index_t *icompq, index_t *nl, index_t *nr, index_t *sqre, index_t &nrhs, float *b, index_t &ldb, float *bx, index_t &ldbx, index_t *perm, index_t *givptr, index_t *givcol, index_t &ldgcol, float *givnum, index_t &ldgnum, float *poles, float *difl, float *difr, float *z__, index_t &k, float *c__, float *s, float *work, index_t *info);
//  static inline void lalsa(index_t *icompq, index_t *smlsiz, index_t &n, index_t &nrhs, float *b, index_t &ldb, float *bx, index_t &ldbx, float *u, index_t &ldu, float *vt, index_t &k, float *difl, float *difr, float *z__, float *poles, index_t *givptr, index_t *givcol, index_t &ldgcol, index_t *perm, float *givnum, float *c__, float *s, float *work, index_t *iwork, index_t *info);
//  static inline void lalsd(const char *uplo, index_t *smlsiz, index_t &n, index_t &nrhs, float *d__, float *e, float *b, index_t &ldb, float *rcond, index_t *rank, float *work, index_t *iwork, index_t *info);
//  static inline void lamc1(index_t *beta, index_t *t, f77_logical *rnd, f77_logical *ieee1);
//  static inline void lamc2(index_t *beta, index_t *t, f77_logical *rnd, float *eps, index_t *emin, float *rmin, index_t *emax, float *rmax);
//  static inline void lamc4(index_t *emin, float *start, index_t *base);
//  static inline void lamc5(index_t *beta, index_t &p, index_t *emin, f77_logical *ieee, index_t *emax, float *rmax);
//  static inline void lamrg(index_t *n1, index_t *n2, float *a, index_t *dtrd1, index_t *dtrd2, index_t *index);
//  static inline void lanv2(float *a, float *b, float *c__, float *d__, float *rt1r, float *rt1i, float *rt2r, float *rt2i, float *cs, float *sn);
//  static inline void lapll(index_t &n, float *x, index_t &incx, float *y, index_t &incy, float *ssmin);
//  static inline void lapmt(f77_logical *forwrd, index_t &m, index_t &n, float *x, index_t &ldx, index_t &k);
//  static inline void laqgb(index_t &m, index_t &n, index_t *kl, index_t *ku, float *ab, index_t &ldab, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, const char *equed);
//  static inline void laqge(index_t &m, index_t &n, float *a, index_t &lda, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, const char *equed);
//  static inline void laqp2(index_t &m, index_t &n, index_t *offset, float *a, index_t &lda, index_t *jpvt, float *tau, float *vn1, float *vn2, float *work);
//  static inline void laqps(index_t &m, index_t &n, index_t *offset, index_t *nb, index_t *kb, float *a, index_t &lda, index_t *jpvt, float *tau, float *vn1, float *vn2, float *auxv, float *f, index_t &ldf);
//  static inline void laqsb(const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqsp(const char *uplo, index_t &n, float *ap, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqsy(const char *uplo, index_t &n, float *a, index_t &lda, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqtr(f77_logical *ltran, f77_logical *lf77_real, index_t &n, float *t, index_t &ldt, float *b, float *w, float *scale, float *x, float *work, index_t *info);
//  static inline void lar1v(index_t &n, index_t *b1, index_t *bn, float *sigma, float *d__, float *l, float *ld, float *lld, float *gersch, float *z__, float *ztz, float *mingma, index_t *r__, index_t *isuppz, float *work);
//  static inline void lar2v(index_t &n, float *x, float *y, float *z__, index_t &incx, float *c__, float *s, index_t &incc);
//  static inline void larf(const char *side, index_t &m, index_t &n, float *v, index_t &incv, float *tau, float *c__, index_t &ldc, float *work);
//  static inline void larfb(const char *side, const char *trans, const char *direct, const char *storev, index_t &m, index_t &n, index_t &k, float *v, index_t &ldv, float *t, index_t &ldt, float *c__, index_t &ldc, float *work, index_t &ldwork);
//  static inline void larfg(index_t &n, float *alpha, float *x, index_t &incx, float *tau);
//  static inline void larft(const char *direct, const char *storev, index_t &n, index_t &k, float *v, index_t &ldv, float *tau, float *t, index_t &ldt);
//  static inline void larfx(const char *side, index_t &m, index_t &n, float *v, float *tau, float *c__, index_t &ldc, float *work);
//  static inline void largv(index_t &n, float *x, index_t &incx, float *y, index_t &incy, float *c__, index_t &incc);
//  static inline void larnv(index_t *idist, index_t *iseed, index_t &n, float *x);
//  static inline void larrb(index_t &n, float *d__, float *l, float *ld, float *lld, index_t *ifirst, index_t *ilast, float *sigma, float *reltol, float *w, float *wgap, float *werr, float *work, index_t *iwork, index_t *info);
//  static inline void larre(index_t &n, float *d__, float *e, float *tol, index_t *nsplit, index_t *isplit, index_t *m_out, float *w, float *woff, float *gersch, float *work, index_t *info);
//  static inline void larrf(index_t &n, float *d__, float *l, float *ld, float *lld, index_t *ifirst, index_t *ilast, float *w, float *dplus, float *lplus, float *work, index_t *iwork, index_t *info);
//  static inline void larrv(index_t &n, float *d__, float *l, index_t *isplit, index_t &m, float *w, index_t *iblock, float *gersch, float *tol, float *z__, index_t &ldz, index_t *isuppz, float *work, index_t *iwork, index_t *info);
//  static inline void lartg(float *f, float *g, float *cs, float *sn, float *r__);
//  static inline void lartv(index_t &n, float *x, index_t &incx, float *y, index_t &incy, float *c__, float *s, index_t &incc);
//  static inline void laruv(index_t *iseed, index_t &n, float *x);
//  static inline void larz(const char *side, index_t &m, index_t &n, index_t *l, float *v, index_t &incv, float *tau, float *c__, index_t &ldc, float *work);
//  static inline void larzb(const char *side, const char *trans, const char *direct, const char *storev, index_t &m, index_t &n, index_t &k, index_t *l, float *v, index_t &ldv, float *t, index_t &ldt, float *c__, index_t &ldc, float *work, index_t &ldwork);
//  static inline void larzt(const char *direct, const char *storev, index_t &n, index_t &k, float *v, index_t &ldv, float *tau, float *t, index_t &ldt);
//  static inline void las2(float *f, float *g, float *h__, float *ssmin, float *ssmax);
//  static inline void lascl(const char *type__, index_t *kl, index_t *ku, float *cfrom, float *cto, index_t &m, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void lasd0(index_t &n, index_t *sqre, float *d__, float *e, float *u, index_t &ldu, float *vt, index_t &ldvt, index_t *smlsiz, index_t *iwork, float *work, index_t *info);
//  static inline void lasd1(index_t *nl, index_t *nr, index_t *sqre, float *d__, float *alpha, float *beta, float *u, index_t &ldu, float *vt, index_t &ldvt, index_t *idxq, index_t *iwork, float *work, index_t *info);
//  static inline void lasd2(index_t *nl, index_t *nr, index_t *sqre, index_t &k, float *d__, float *z__, float *alpha, float *beta, float *u, index_t &ldu, float *vt, index_t &ldvt, float *dsigma, float *u2, index_t &ldu2, float *vt2, index_t &ldvt2, index_t *idxp, index_t *idx, index_t *idxc, index_t *idxq, index_t *coltyp, index_t *info);
//  static inline void lasd3(index_t *nl, index_t *nr, index_t *sqre, index_t &k, float *d__, float *q, index_t &ldq, float *dsigma, float *u, index_t &ldu, float *u2, index_t &ldu2, float *vt, index_t &ldvt, float *vt2, index_t &ldvt2, index_t *idxc, index_t *ctot, float *z__, index_t *info);
//  static inline void lasd4(index_t &n, index_t *i__, float *d__, float *z__, float *delta, float *rho, float *sigma, float *work, index_t *info);
//  static inline void lasd5(index_t *i__, float *d__, float *z__, float *delta, float *rho, float *dsigma, float *work);
//  static inline void lasd6(index_t *icompq, index_t *nl, index_t *nr, index_t *sqre, float *d__, float *vf, float *vl, float *alpha, float *beta, index_t *idxq, index_t *perm, index_t *givptr, index_t *givcol, index_t &ldgcol, float *givnum, index_t &ldgnum, float *poles, float *difl, float *difr, float *z__, index_t &k, float *c__, float *s, float *work, index_t *iwork, index_t *info);
//  static inline void lasd7(index_t *icompq, index_t *nl, index_t *nr, index_t *sqre, index_t &k, float *d__, float *z__, float *zw, float *vf, float *vfw, float *vl, float *vlw, float *alpha, float *beta, float *dsigma, index_t *idx, index_t *idxp, index_t *idxq, index_t *perm, index_t *givptr, index_t *givcol, index_t &ldgcol, float *givnum, index_t &ldgnum, float *c__, float *s, index_t *info);
//  static inline void lasd8(index_t *icompq, index_t &k, float *d__, float *z__, float *vf, float *vl, float *difl, float *difr, index_t &lddifr, float *dsigma, float *work, index_t *info);
//  static inline void lasd9(index_t *icompq, index_t &ldu, index_t &k, float *d__, float *z__, float *vf, float *vl, float *difl, float *difr, float *dsigma, float *work, index_t *info);
//  static inline void lasda(index_t *icompq, index_t *smlsiz, index_t &n, index_t *sqre, float *d__, float *e, float *u, index_t &ldu, float *vt, index_t &k, float *difl, float *difr, float *z__, float *poles, index_t *givptr, index_t *givcol, index_t &ldgcol, index_t *perm, float *givnum, float *c__, float *s, float *work, index_t *iwork, index_t *info);
//  static inline void lasdq(const char *uplo, index_t *sqre, index_t &n, index_t &ncvt, index_t &nru, index_t &ncc, float *d__, float *e, float *vt, index_t &ldvt, float *u, index_t &ldu, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void lasdt(index_t &n, index_t *lvl, index_t *nd, index_t *inode, index_t *ndiml, index_t *ndimr, index_t *msub);
//  static inline void laset(const char *uplo, index_t &m, index_t &n, float *alpha, float *beta, float *a, index_t &lda);
//  static inline void lasq1(index_t &n, float *d__, float *e, float *work, index_t *info);
//  static inline void lasq2(index_t &n, float *z__, index_t *info);
//  static inline void lasq3(index_t *i0, index_t *n0, float *z__, index_t *pp, float *dmin__, float *sigma, float *desig, float *qmax, index_t *nfail, index_t *iter, index_t *ndiv, f77_logical *ieee);
//  static inline void lasq4(index_t *i0, index_t *n0, float *z__, index_t *pp, index_t *n0in, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dn1, float *dn2, float *tau, index_t *ttype);
//  static inline void lasq5(index_t *i0, index_t *n0, float *z__, index_t *pp, float *tau, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dnm1, float *dnm2, f77_logical *ieee);
//  static inline void lasq6(index_t *i0, index_t *n0, float *z__, index_t *pp, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dnm1, float *dnm2);
//  static inline void lasr(const char *side, const char *pivot, const char *direct, index_t &m, index_t &n, float *c__, float *s, float *a, index_t &lda);
//  static inline void lasrt(const char *id, index_t &n, float *d__, index_t *info);
//  static inline void lassq(index_t &n, float *x, index_t &incx, float *scale, float *sumsq);
//  static inline void lasv2(float *f, float *g, float *h__, float *ssmin, float *ssmax, float *snr, float *csr, float *snl, float *csl);
//  static inline void laswp(index_t &n, float *a, index_t &lda, index_t *k1, index_t *k2, index_t *ipiv, index_t &incx);
//  static inline void lasy2(f77_logical *ltranl, f77_logical *ltranr, index_t *isgn, index_t *n1, index_t *n2, float *tl, index_t &ldtl, float *tr, index_t &ldtr, float *b, index_t &ldb, float *scale, float *x, index_t &ldx, float *xnorm, index_t *info);
//  static inline void lasyf(const char *uplo, index_t &n, index_t *nb, index_t *kb, float *a, index_t &lda, index_t *ipiv, float *w, index_t &ldw, index_t *info);
//  static inline void latbs(const char *uplo, const char *trans, const char *diag, const char *normin, index_t &n, index_t *kd, float *ab, index_t &ldab, float *x, float *scale, float *cnorm, index_t *info);
//  static inline void latdf(index_t *ijob, index_t &n, float *z__, index_t &ldz, float *rhs, float *rdsum, float *rdscal, index_t *ipiv, index_t *jpiv);
//  static inline void latps(const char *uplo, const char *trans, const char *diag, const char *normin, index_t &n, float *ap, float *x, float *scale, float *cnorm, index_t *info);
//  static inline void latrd(const char *uplo, index_t &n, index_t *nb, float *a, index_t &lda, float *e, float *tau, float *w, index_t &ldw);
//  static inline void latrs(const char *uplo, const char *trans, const char *diag, const char *normin, index_t &n, float *a, index_t &lda, float *x, float *scale, float *cnorm, index_t *info);
//  static inline void latrz(index_t &m, index_t &n, index_t *l, float *a, index_t &lda, float *tau, float *work);
//  static inline void latzm(const char *side, index_t &m, index_t &n, float *v, index_t &incv, float *tau, float *c1, float *c2, index_t &ldc, float *work);
//  static inline void lauu2(const char *uplo, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void lauum(const char *uplo, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void opgtr(const char *uplo, index_t &n, float *ap, float *tau, float *q, index_t &ldq, float *work, index_t *info);
//  static inline void opmtr(const char *side, const char *uplo, const char *trans, index_t &m, index_t &n, float *ap, float *tau, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void org2l(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void org2r(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void orgbr(const char *vect, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void orghr(index_t &n, index_t *ilo, index_t *ihi, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void orgl2(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void orglq(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void orgql(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
  static inline void orgqr(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(orgqr)(m, n, k, a, lda, tau, work, lwork, info);

  }
//  static inline void orgr2(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t *info);
//  static inline void orgrq(index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void orgtr(const char *uplo, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void orm2l(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void orm2r(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void ormbr(const char *vect, const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void ormhr(const char *side, const char *trans, index_t &m, index_t &n, index_t *ilo, index_t *ihi, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void orml2(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void ormlq(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void ormql(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void ormqr(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void ormr2(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void ormr3(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, index_t *l, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t *info);
//  static inline void ormrq(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void ormrz(const char *side, const char *trans, index_t &m, index_t &n, index_t &k, index_t *l, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void ormtr(const char *side, const char *uplo, const char *trans, index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *c__, index_t &ldc, float *work, index_t &lwork, index_t *info);
//  static inline void pbcon(const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void pbequ(const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *s, float *scond, float *amax, index_t *info);
//  static inline void pbrfs(const char *uplo, index_t &n, index_t *kd, index_t &nrhs, float *ab, index_t &ldab, float *afb, index_t &ldafb, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void pbstf(const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, index_t *info);
//  static inline void pbsv(const char *uplo, index_t &n, index_t *kd, index_t &nrhs, float *ab, index_t &ldab, float *b, index_t &ldb, index_t *info);
//  static inline void pbsvx(const char *fact, const char *uplo, index_t &n, index_t *kd, index_t &nrhs, float *ab, index_t &ldab, float *afb, index_t &ldafb, const char *equed, float *s, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void pbtf2(const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, index_t *info);
//  static inline void pbtrf(const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, index_t *info);
//  static inline void pbtrs(const char *uplo, index_t &n, index_t *kd, index_t &nrhs, float *ab, index_t &ldab, float *b, index_t &ldb, index_t *info);
//  static inline void pocon(const char *uplo, index_t &n, float *a, index_t &lda, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void poequ(index_t &n, float *a, index_t &lda, float *s, float *scond, float *amax, index_t *info);
//  static inline void porfs(const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, float *af, index_t &ldaf, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void posv(const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, index_t *info);
//  static inline void posvx(const char *fact, const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, float *af, index_t &ldaf, const char *equed, float *s, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void potf2(const char *uplo, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void potrf(const char *uplo, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void potri(const char *uplo, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void potrs(const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, index_t *info);
//  static inline void ppcon(const char *uplo, index_t &n, float *ap, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void ppequ(const char *uplo, index_t &n, float *ap, float *s, float *scond, float *amax, index_t *info);
//  static inline void pprfs(const char *uplo, index_t &n, index_t &nrhs, float *ap, float *afp, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void ppsv(const char *uplo, index_t &n, index_t &nrhs, float *ap, float *b, index_t &ldb, index_t *info);
//  static inline void ppsvx(const char *fact, const char *uplo, index_t &n, index_t &nrhs, float *ap, float *afp, const char *equed, float *s, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void pptrf(const char *uplo, index_t &n, float *ap, index_t *info);
//  static inline void pptri(const char *uplo, index_t &n, float *ap, index_t *info);
//  static inline void pptrs(const char *uplo, index_t &n, index_t &nrhs, float *ap, float *b, index_t &ldb, index_t *info);
//  static inline void ptcon(index_t &n, float *d__, float *e, float *anorm, float *rcond, float *work, index_t *info);
//  static inline void pteqr(const char *compz, index_t &n, float *d__, float *e, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void ptrfs(index_t &n, index_t &nrhs, float *d__, float *e, float *df, float *ef, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *info);
//  static inline void ptsv(index_t &n, index_t &nrhs, float *d__, float *e, float *b, index_t &ldb, index_t *info);
//  static inline void ptsvx(const char *fact, index_t &n, index_t &nrhs, float *d__, float *e, float *df, float *ef, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *info);
//  static inline void pttrf(index_t &n, float *d__, float *e, index_t *info);
//  static inline void pttrs(index_t &n, index_t &nrhs, float *d__, float *e, float *b, index_t &ldb, index_t *info);
//  static inline void ptts2(index_t &n, index_t &nrhs, float *d__, float *e, float *b, index_t &ldb);
//  static inline void rscl(index_t &n, float *sa, float *sx, index_t &incx);
//  static inline void sbev(const char *jobz, const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *w, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void sbevd(const char *jobz, const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *w, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void sbevx(const char *jobz, const char *range, const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *q, index_t &ldq, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void sbgst(const char *vect, const char *uplo, index_t &n, index_t *ka, index_t *kb, float *ab, index_t &ldab, float *bb, index_t &ldbb, float *x, index_t &ldx, float *work, index_t *info);
//  static inline void sbgv(const char *jobz, const char *uplo, index_t &n, index_t *ka, index_t *kb, float *ab, index_t &ldab, float *bb, index_t &ldbb, float *w, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void sbgvd(const char *jobz, const char *uplo, index_t &n, index_t *ka, index_t *kb, float *ab, index_t &ldab, float *bb, index_t &ldbb, float *w, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void sbgvx(const char *jobz, const char *range, const char *uplo, index_t &n, index_t *ka, index_t *kb, float *ab, index_t &ldab, float *bb, index_t &ldbb, float *q, index_t &ldq, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void sbtrd(const char *vect, const char *uplo, index_t &n, index_t *kd, float *ab, index_t &ldab, float *d__, float *e, float *q, index_t &ldq, float *work, index_t *info);
//  static inline void spcon(const char *uplo, index_t &n, float *ap, index_t *ipiv, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void spev(const char *jobz, const char *uplo, index_t &n, float *ap, float *w, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void spevd(const char *jobz, const char *uplo, index_t &n, float *ap, float *w, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void spevx(const char *jobz, const char *range, const char *uplo, index_t &n, float *ap, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void spgst(index_t *itype, const char *uplo, index_t &n, float *ap, float *bp, index_t *info);
//  static inline void spgv(index_t *itype, const char *jobz, const char *uplo, index_t &n, float *ap, float *bp, float *w, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void spgvd(index_t *itype, const char *jobz, const char *uplo, index_t &n, float *ap, float *bp, float *w, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void spgvx(index_t *itype, const char *jobz, const char *range, const char *uplo, index_t &n, float *ap, float *bp, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void sprfs(const char *uplo, index_t &n, index_t &nrhs, float *ap, float *afp, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void spsv(const char *uplo, index_t &n, index_t &nrhs, float *ap, index_t *ipiv, float *b, index_t &ldb, index_t *info);
//  static inline void spsvx(const char *fact, const char *uplo, index_t &n, index_t &nrhs, float *ap, float *afp, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void sptrd(const char *uplo, index_t &n, float *ap, float *d__, float *e, float *tau, index_t *info);
//  static inline void sptrf(const char *uplo, index_t &n, float *ap, index_t *ipiv, index_t *info);
//  static inline void sptri(const char *uplo, index_t &n, float *ap, index_t *ipiv, float *work, index_t *info);
//  static inline void sptrs(const char *uplo, index_t &n, index_t &nrhs, float *ap, index_t *ipiv, float *b, index_t &ldb, index_t *info);
//  static inline void stebz(const char *range, const char *order, index_t &n, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, float *d__, float *e, index_t *m_out, index_t *nsplit, float *w, index_t *iblock, index_t *isplit, float *work, index_t *iwork, index_t *info);
//  static inline void stedc(const char *compz, index_t &n, float *d__, float *e, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void stegr(const char *jobz, const char *range, index_t &n, float *d__, float *e, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, index_t *isuppz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void stein(index_t &n, float *d__, float *e, index_t &m, float *w, index_t *iblock, index_t *isplit, float *z__, index_t &ldz, float *work, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void steqr(const char *compz, index_t &n, float *d__, float *e, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void sterf(index_t &n, float *d__, float *e, index_t *info);
//  static inline void stev(const char *jobz, index_t &n, float *d__, float *e, float *z__, index_t &ldz, float *work, index_t *info);
//  static inline void stevd(const char *jobz, index_t &n, float *d__, float *e, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void stevr(const char *jobz, const char *range, index_t &n, float *d__, float *e, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, index_t *isuppz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void stevx(const char *jobz, const char *range, index_t &n, float *d__, float *e, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void sycon(const char *uplo, index_t &n, float *a, index_t &lda, index_t *ipiv, float *anorm, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void syev(const char *jobz, const char *uplo, index_t &n, float *a, index_t &lda, float *w, float *work, index_t &lwork, index_t *info);
//  static inline void syevd(const char *jobz, const char *uplo, index_t &n, float *a, index_t &lda, float *w, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void syevr(const char *jobz, const char *range, const char *uplo, index_t &n, float *a, index_t &lda, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, index_t *isuppz, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void syevx(const char *jobz, const char *range, const char *uplo, index_t &n, float *a, index_t &lda, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void sygs2(index_t *itype, const char *uplo, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, index_t *info);
//  static inline void sygst(index_t *itype, const char *uplo, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, index_t *info);
  static inline void sygv(index_t *itype, const char *jobz, const char *uplo, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *w, float *work, index_t &lwork, index_t *info) {
    inline F77_FUNC(sygv)(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, *info);

  }
//  static inline void sygvd(index_t *itype, const char *jobz, const char *uplo, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *w, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void sygvx(index_t *itype, const char *jobz, const char *range, const char *uplo, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *vl, float *vu, index_t *il, index_t *iu, float *abstol, index_t *m_out, float *w, float *z__, index_t &ldz, float *work, index_t &lwork, index_t *iwork, index_t *ifail, index_t *info);
//  static inline void syrfs(const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, float *af, index_t &ldaf, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void sysv(const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, index_t *ipiv, float *b, index_t &ldb, float *work, index_t &lwork, index_t *info);
//  static inline void sysvx(const char *fact, const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, float *af, index_t &ldaf, index_t *ipiv, float *b, index_t &ldb, float *x, index_t &ldx, float *rcond, float *ferr, float *berr, float *work, index_t &lwork, index_t *iwork, index_t *info);
//  static inline void sytd2(const char *uplo, index_t &n, float *a, index_t &lda, float *d__, float *e, float *tau, index_t *info);
//  static inline void sytf2(const char *uplo, index_t &n, float *a, index_t &lda, index_t *ipiv, index_t *info);
//  static inline void sytrd(const char *uplo, index_t &n, float *a, index_t &lda, float *d__, float *e, float *tau, float *work, index_t &lwork, index_t *info);
//  static inline void sytrf(const char *uplo, index_t &n, float *a, index_t &lda, index_t *ipiv, float *work, index_t &lwork, index_t *info);
//  static inline void sytri(const char *uplo, index_t &n, float *a, index_t &lda, index_t *ipiv, float *work, index_t *info);
//  static inline void sytrs(const char *uplo, index_t &n, index_t &nrhs, float *a, index_t &lda, index_t *ipiv, float *b, index_t &ldb, index_t *info);
//  static inline void tbcon(const char *norm, const char *uplo, const char *diag, index_t &n, index_t *kd, float *ab, index_t &ldab, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void tbrfs(const char *uplo, const char *trans, const char *diag, index_t &n, index_t *kd, index_t &nrhs, float *ab, index_t &ldab, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void tbtrs(const char *uplo, const char *trans, const char *diag, index_t &n, index_t *kd, index_t &nrhs, float *ab, index_t &ldab, float *b, index_t &ldb, index_t *info);
//  static inline void tgevc(const char *side, const char *howmny, f77_logical *select, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *vl, index_t &ldvl, float *vr, index_t &ldvr, index_t *mm, index_t *m_out, float *work, index_t *info);
//  static inline void tgex2(f77_logical &wantq, f77_logical &wantz, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *q, index_t &ldq, float *z__, index_t &ldz, index_t *j1, index_t *n1, index_t *n2, float *work, index_t &lwork, index_t *info);
//  static inline void tgexc(f77_logical &wantq, f77_logical &wantz, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *q, index_t &ldq, float *z__, index_t &ldz, index_t *ifst, index_t *ilst, float *work, index_t &lwork, index_t *info);
//  static inline void tgsen(index_t *ijob, f77_logical &wantq, f77_logical &wantz, f77_logical *select, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *alphar, float *alphai, float *beta, float *q, index_t &ldq, float *z__, index_t &ldz, index_t *m_out, float *pl, float *pr, float *dif, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void tgsja(const char *jobu, const char *jobv, const char *jobq, index_t &m, index_t &p, index_t &n, index_t &k, index_t *l, float *a, index_t &lda, float *b, index_t &ldb, float *tola, float *tolb, float *alpha, float *beta, float *u, index_t &ldu, float *v, index_t &ldv, float *q, index_t &ldq, float *work, index_t *ncycle, index_t *info);
//  static inline void tgsna(const char *job, const char *howmny, f77_logical *select, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *vl, index_t &ldvl, float *vr, index_t &ldvr, float *s, float *dif, index_t *mm, index_t *m_out, float *work, index_t &lwork, index_t *iwork, index_t *info);
//  static inline void tgsy2(const char *trans, index_t *ijob, index_t &m, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *c__, index_t &ldc, float *d__, index_t &ldd, float *e, index_t &lde, float *f, index_t &ldf, float *scale, float *rdsum, float *rdscal, index_t *iwork, index_t *pq, index_t *info);
//  static inline void tgsyl(const char *trans, index_t *ijob, index_t &m, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *c__, index_t &ldc, float *d__, index_t &ldd, float *e, index_t &lde, float *f, index_t &ldf, float *scale, float *dif, float *work, index_t &lwork, index_t *iwork, index_t *info);
//  static inline void tpcon(const char *norm, const char *uplo, const char *diag, index_t &n, float *ap, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void tprfs(const char *uplo, const char *trans, const char *diag, index_t &n, index_t &nrhs, float *ap, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void tptri(const char *uplo, const char *diag, index_t &n, float *ap, index_t *info);
//  static inline void tptrs(const char *uplo, const char *trans, const char *diag, index_t &n, index_t &nrhs, float *ap, float *b, index_t &ldb, index_t *info);
//  static inline void trcon(const char *norm, const char *uplo, const char *diag, index_t &n, float *a, index_t &lda, float *rcond, float *work, index_t *iwork, index_t *info);
//  static inline void trevc(const char *side, const char *howmny, f77_logical *select, index_t &n, float *t, index_t &ldt, float *vl, index_t &ldvl, float *vr, index_t &ldvr, index_t *mm, index_t *m_out, float *work, index_t *info);
//  static inline void trexc(const char *compq, index_t &n, float *t, index_t &ldt, float *q, index_t &ldq, index_t *ifst, index_t *ilst, float *work, index_t *info);
//  static inline void trrfs(const char *uplo, const char *trans, const char *diag, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, float *x, index_t &ldx, float *ferr, float *berr, float *work, index_t *iwork, index_t *info);
//  static inline void trsen(const char *job, const char *compq, f77_logical *select, index_t &n, float *t, index_t &ldt, float *q, index_t &ldq, float *wr, float *wi, index_t *m_out, float *s, float *sep, float *work, index_t &lwork, index_t *iwork, index_t &liwork, index_t *info);
//  static inline void trsna(const char *job, const char *howmny, f77_logical *select, index_t &n, float *t, index_t &ldt, float *vl, index_t &ldvl, float *vr, index_t &ldvr, float *s, float *sep, index_t *mm, index_t *m_out, float *work, index_t &ldwork, index_t *iwork, index_t *info);
//  static inline void trsyl(const char *trana, const char *tranb, index_t *isgn, index_t &m, index_t &n, float *a, index_t &lda, float *b, index_t &ldb, float *c__, index_t &ldc, float *scale, index_t *info);
//  static inline void trti2(const char *uplo, const char *diag, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void trtri(const char *uplo, const char *diag, index_t &n, float *a, index_t &lda, index_t *info);
//  static inline void trtrs(const char *uplo, const char *trans, const char *diag, index_t &n, index_t &nrhs, float *a, index_t &lda, float *b, index_t &ldb, index_t *info);
//  static inline void tzrqf(index_t &m, index_t &n, float *a, index_t &lda, float *tau, index_t *info);
//  static inline void tzrzf(index_t &m, index_t &n, float *a, index_t &lda, float *tau, float *work, index_t &lwork, index_t *info);
//  inline index_t icmax1(index_t &n, f77_complex *cx, index_t &incx);
//  inline index_t ieeeck(index_t &ispec, f77_real *zero, f77_real *one);
//  inline index_t ilaenv(index_t &ispec, const char *name__, const char *opts, index_t *n1, index_t *n2, index_t *n3, index_t *n4, f77_str_len name_len, f77_str_len opts_len);
//  inline index_t izmax1(index_t &n, floatcomplex *cx, index_t &incx);

};

static CppLapack<float> cpplapack_float;

};
