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
class CppLapack {
};

template<>
class CppLapack<float> {
 public:
  CppLapack() {
    float fake_matrix[64];
    float fake_workspace;
    float fake_vector;
    f77_integer fake_pivots;
    f77_integer fake_info;
      
    /* TODO: This may want to be ilaenv */
    this->getri(1, (float *)fake_matrix, 1, &fake_pivots, &fake_workspace,
        -1, &fake_info);
    this->getri_block_size = int(fake_workspace);
      
    this->geqrf(1, 1, (float *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
        &fake_info);
    this->geqrf_block_size = int(fake_workspace);
      
    this->orgqr(1, 1, 1, (float *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
         &fake_info);
    this->orgqr_block_size = int(fake_workspace);
      
    this->geqrf_dorgqr_block_size =
         std::max(this->geqrf_block_size, this->orgqr_block_size);
 
  }
  static int getri_block_size;
  static int geqrf_block_size;
  static int orgqr_block_size;
  static int geqrf_dorgqr_block_size;


  static inline void bdsdc(const char *uplo, const char *compq, f77_integer &n, float *d__, float *e, float *u, f77_integer &ldu, float *vt, f77_integer &ldvt, float *q, f77_integer *iq, float *work, f77_integer *iwork, f77_integer *info) {
     F77_FUNC(sbdsdc)(uplo, compq, n, d__, e, u, ldu, vt, ldvt, q, iq, work, iwork, info); 
  }
  static inline void bdsqr(const char *uplo, f77_integer &n, f77_integer &ncvt, f77_integer &nru, f77_integer &ncc, float *d__, float *e, float *vt, f77_integer &ldvt, float *u, f77_integer &ldu, float *c__, f77_integer &ldc, float *work, f77_integer *info) {
     F77_FUNC(sbdsqr)(uplo, n, ncvt, nru, ncc, d__, e, vt, ldvt, u, ldu, c__, ldc, work, info);
  }
  static inline void disna(const char *job, f77_integer &m, f77_integer &n, float *d__, float *sep, f77_integer *info) {
     F77_FUNC(sdisna)(job, m, n, d__, sep, info);
  }
  static inline void gbbrd(const char *vect, f77_integer &m, f77_integer &n, f77_integer &ncc, f77_integer *kl, f77_integer *ku, float *ab, f77_integer &ldab, float *d__, float *e, float *q, f77_integer &ldq, float *pt, f77_integer &ldpt, float *c__, f77_integer &ldc, float *work, f77_integer *info) {
     F77_FUNC(sgbbrd)(vect, m, n, ncc, kl, ku, ab, ldab, d__, e, q, ldq, pt, ldpt, c__, ldc, work, info);
  }
  static inline void gbcon(const char *norm, f77_integer &n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer &ldab, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info) {
     F77_FUNC(sgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork, info);
  }
  static inline void gbequ(f77_integer &m, f77_integer &n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer &ldab, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, f77_integer *info) {
     F77_FUNC(sgbequ)(m, n, kl, ku, ab, ldab, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gbrfs(const char *trans, f77_integer &n, f77_integer *kl, f77_integer *ku, f77_integer &nrhs, float *ab, f77_integer &ldab, float *afb, f77_integer &ldafb, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer*iwork, f77_integer *info) {
    F77_FUNC(sgbrfs)(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  }
  static inline void gbsv(f77_integer &n, f77_integer *kl, f77_integer *ku, f77_integer &nrhs, float *ab, f77_integer &ldab, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info) {
    F77_FUNC(sgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
  }
  static inline void gbsvx(const char *fact, const char *trans, f77_integer &n, f77_integer *kl, f77_integer *ku, f77_integer &nrhs, float *ab, f77_integer &ldab, float *afb, f77_integer &ldafb, f77_integer *ipiv, const char *equed, float *r__, float *c__, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r__, c__, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
  }
  static inline void gbtf2(f77_integer &m, f77_integer &n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer &ldab, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(sgbtf2)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrf(f77_integer &m, f77_integer &n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer &ldab, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(sgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrs(const char *trans, f77_integer &n, f77_integer *kl, f77_integer *ku, f77_integer &nrhs, float *ab, f77_integer &ldab, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info) {
    F77_FUNC(sgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);

  }
  static inline void gebak(const char *job, const char *side, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *scale, f77_integer &m, float *v, f77_integer &ldv, f77_integer *info) {
    F77_FUNC(sgebak)(job, side, n, ilo, ihi, scale, m, v, ldv, info);
  }
  static inline void gebal(const char *job, f77_integer &n, float *a, f77_integer &lda, f77_integer *ilo, f77_integer *ihi, float *scale, f77_integer *info) {
    F77_FUNC(sgebal)(job, n, a, lda, ilo, ihi, scale, info);

  }
  static inline void gebd2(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *d__, float *e, float *tauq, float *taup, float *work, f77_integer *info) {
    F77_FUNC(sgebd2)(m, n, a, lda, d__, e, tauq, taup, work, info);
  }
  static inline void gebrd(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *d__, float *e, float *tauq, float *taup, float *work, f77_integer &lwork, f77_integer *info) {
    F77_FUNC(sgebrd)(m, n, a, lda, d__, e, tauq, taup, work, lwork, info);
  }
  static inline void gecon(const char *norm, f77_integer &n, float *a, f77_integer &lda, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
  }
  static inline void geequ(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, f77_integer *info) {
    F77_FUNC(sgeequ)(m, n, a, lda, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gees(const char *jobvs, const char *sort, f77_logical_func select, f77_integer &n, float *a, f77_integer &lda, f77_integer *sdim, float *wr, float *wi, float *vs, f77_integer &ldvs, float *work, f77_integer &lwork, unsigned int *bwork, f77_integer *info) {
    F77_FUNC(sgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
  }
  static inline void geesx(const char *jobvs, const char *sort, f77_logical_func select, const char *sense, f77_integer &n, float *a, f77_integer &lda, f77_integer *sdim, float *wr, float *wi, float *vs, f77_integer &ldvs, float *rconde, float *rcondv, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, unsigned int *bwork, f77_integer *info) {
    F77_FUNC(sgeesx)(jobvs, sort, select, sense, n, a, lda, sdim, wr, wi, vs, ldvs, rconde, rcondv, work, lwork, iwork, liwork, bwork, info);
  }
  static inline void geev(const char *jobvl, const char *jobvr, f77_integer &n, float *a, f77_integer &lda, float *wr, float *wi, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, float *work, f77_integer &lwork, f77_integer *info) {
    F77_FUNC(sgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
  }
  static inline void geevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, f77_integer &n, float *a, f77_integer &lda, float *wr, float *wi, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, f77_integer *ilo, f77_integer *ihi, float *scale, float *abnrm, float *rconde, float *rcondv, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work,lwork, iwork, info);
  }
  static inline void gegs(const char *jobvsl, const char *jobvsr, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *vsl, f77_integer &ldvsl, float *vsr, f77_integer &ldvsr, float *work, f77_integer &lwork, f77_integer *info) {
    F77_FUNC(sgegs)(jobvsl, jobvsr, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, info);
  }
  static inline void gegv(const char *jobvl, const char *jobvr, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, float *work, f77_integer &lwork, f77_integer *info) {
    F77_FUNC(sgegv)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
  }
//  static inline void gehd2(f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void gehrd(f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gelq2(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void gelqf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gels(const char *trans, f77_integer &m, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gelsd(f77_integer &m, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *s, float *rcond, f77_integer *rank, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *info);
//  static inline void gelss(f77_integer &m, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *s, float *rcond, f77_integer *rank, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gelsx(f77_integer &m, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *jpvt, float *rcond, f77_integer *rank, float *work, f77_integer *info);
//  static inline void gelsy(f77_integer &m, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *jpvt, float *rcond, f77_integer *rank, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void geql2(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void geqlf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
// static inline void geqp3(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, f77_integer *jpvt, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void geqpf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, f77_integer *jpvt, float *tau, float *work, f77_integer *info);
//  static inline void geqr2(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
  static inline void geqrf(f77_integer m, f77_integer n, float *a, f77_integer lda, float *tau, float *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(sgeqrf)(m, n, a, lda, tau, work, lwork, info);
  }
  static inline void gerfs(const char *trans, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *af, f77_integer &ldaf, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgerfs)(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  
  }
//  static inline void gerq2(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void gerqf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gesc2(f77_integer &n, float *a, f77_integer &lda, float *rhs, f77_integer *ipiv, f77_integer *jpiv, float *scale);
  static inline void gesdd(const char *jobz, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *s, float *u, f77_integer &ldu, float *vt, f77_integer &ldvt, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static inline void gesv(f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info) {
    F77_FUNC(sgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
  }
//  static inline void gesvd(const char *jobu, const char *jobvt, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *s, float *u, f77_integer &ldu, float *vt, f77_integer &ldvt, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gesvx(const char *fact, const char *trans, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *af, f77_integer &ldaf, f77_integer *ipiv, const char *equed, float *r__, float *c__, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void getc2(f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, f77_integer *jpiv, f77_integer *info);
//  static inline void getf2(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, f77_integer *info);
  static inline void getrf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(sgetrf)(m, n, a, lda, ipiv, info); 
  }
  static inline void getri(f77_integer n, float *a, f77_integer lda, f77_integer *ipiv, float *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(sgetri)(n, a, lda, ipiv, work, lwork, info);
  }
//  static inline void getrs(const char *trans, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void ggbak(const char *job, const char *side, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *lscale, float *rscale, f77_integer &m, float *v, f77_integer &ldv, f77_integer *info);
//  static inline void ggbal(const char *job, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *ilo, f77_integer *ihi, float *lscale, float *rscale, float *work, f77_integer *info);
//  static inline void gges(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *sdim, float *alphar, float *alphai, float *beta, float *vsl, f77_integer &ldvsl, float *vsr, f77_integer &ldvsr, float *work, f77_integer &lwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggesx(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, const char *sense, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *sdim, float *alphar, float *alphai, float *beta, float *vsl, f77_integer &ldvsl, float *vsr, f77_integer &ldvsr, float *rconde, float *rcondv, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggev(const char *jobvl, const char *jobvr, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ggevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, f77_integer *ilo, f77_integer *ihi, float *lscale, float *rscale, float *abnrm, float *bbnrm, float *rconde, float *rcondv, float *work, f77_integer &lwork, f77_integer *iwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggglm(f77_integer &n, f77_integer &m, f77_integer &p, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *d__, float *x, float *y, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void gghrd(const char *compq, const char *compz, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *q, f77_integer &ldq, float *z__, f77_integer &ldz, f77_integer *info);
//  static inline void gglse(f77_integer &m, f77_integer &n, f77_integer &p, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *c__, float *d__, float *x, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ggqrf(f77_integer &n, f77_integer &m, f77_integer &p, float *a, f77_integer &lda, float *taua, float *b, f77_integer &ldb, float *taub, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ggrqf(f77_integer &m, f77_integer &p, f77_integer &n, float *a, f77_integer &lda, float *taua, float *b, f77_integer &ldb, float *taub, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ggsvd(const char *jobu, const char *jobv, const char *jobq, f77_integer &m, f77_integer &n, f77_integer &p, f77_integer &k, f77_integer *l, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alpha, float *beta, float *u, f77_integer &ldu, float *v, f77_integer &ldv, float *q, f77_integer &ldq, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void ggsvp(const char *jobu, const char *jobv, const char *jobq, f77_integer &m, f77_integer &p, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *tola, float *tolb, f77_integer &k, f77_integer *l, float *u, f77_integer &ldu, float *v, f77_integer &ldv, float *q, f77_integer &ldq, f77_integer *iwork, float *tau, float *work, f77_integer *info);
//  static inline void gtcon(const char *norm, f77_integer &n, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void gtrfs(const char *trans, f77_integer &n, f77_integer &nrhs, float *dl, float *d__, float *du, float *dlf, float *df, float *duf, float *du2, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void gtsv(f77_integer &n, f77_integer &nrhs, float *dl, float *d__, float *du, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void gtsvx(const char *fact, const char *trans, f77_integer &n, f77_integer &nrhs, float *dl, float *d__, float *du, float *dlf, float *df, float *duf, float *du2, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void gttrf(f77_integer &n, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, f77_integer *info);
//  static inline void gttrs(const char *trans, f77_integer &n, f77_integer &nrhs, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void gtts2(f77_integer *itrans, f77_integer &n, f77_integer &nrhs, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, float *b, f77_integer &ldb);
//  static inline void hgeqz(const char *job, const char *compq, const char *compz, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *q, f77_integer &ldq, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void hsein(const char *side, const char *eigsrc, const char *initv, f77_logical *select, f77_integer &n, float *h__, f77_integer &ldh, float *wr, float *wi, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, f77_integer *mm, f77_integer *m_out, float *work, f77_integer *ifaill, f77_integer *ifailr, f77_integer *info);
//  static inline void hseqr(const char *job, const char *compz, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *h__, f77_integer &ldh, float *wr, float *wi, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void labad(float *small, float *large);
//  static inline void labrd(f77_integer &m, f77_integer &n, f77_integer *nb, float *a, f77_integer &lda, float *d__, float *e, float *tauq, float *taup, float *x, f77_integer &ldx, float *y, f77_integer &ldy);
//  static inline void lacon(f77_integer &n, float *v, float *x, f77_integer *isgn, float *est, f77_integer *kase);
//  static inline void lacpy(const char *uplo, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb);
//  static inline void ladiv(float *a, float *b, float *c__, float *d__, float *p, float *q);
//  static inline void lae2(float *a, float *b, float *c__, float *rt1, float *rt2);
//  static inline void laebz(f77_integer *ijob, f77_integer *nitmax, f77_integer &n, f77_integer *mmax, f77_integer *minp, f77_integer *nbmin, float *abstol, float *reltol, float *pivmin, float *d__, float *e, float *e2, f77_integer *nval, float *ab, float *c__, f77_integer *mout, f77_integer *nab, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed0(f77_integer *icompq, f77_integer *qsiz, f77_integer &n, float *d__, float *e, float *q, f77_integer &ldq, float *qstore, f77_integer &ldqs, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed1(f77_integer &n, float *d__, float *q, f77_integer &ldq, f77_integer *indxq, float *rho, f77_integer *cutpnt, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed2(f77_integer &k, f77_integer &n, f77_integer *n1, float *d__, float *q, f77_integer &ldq, f77_integer *indxq, float *rho, float *z__, float *dlamda, float *w, float *q2, f77_integer *indx, f77_integer *indxc, f77_integer *indxp, f77_integer *coltyp, f77_integer *info);
//  static inline void laed3(f77_integer &k, f77_integer &n, f77_integer *n1, float *d__, float *q, f77_integer &ldq, float *rho, float *dlamda, float *q2, f77_integer *indx, f77_integer *ctot, float *w, float *s, f77_integer *info);
//  static inline void laed4(f77_integer &n, f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *dlam, f77_integer *info);
//  static inline void laed5(f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *dlam);
//  static inline void laed6(f77_integer *kniter, f77_logical *orgati, float *rho, float *d__, float *z__, float *finit, float *tau, f77_integer *info);
//  static inline void laed7(f77_integer *icompq, f77_integer &n, f77_integer *qsiz, f77_integer *tlvls, f77_integer *curlvl, f77_integer *curpbm, float *d__, float *q, f77_integer &ldq, f77_integer *indxq, float *rho, f77_integer *cutpnt, float *qstore, f77_integer *qptr, f77_integer *prmptr, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, float *givnum, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed8(f77_integer *icompq, f77_integer &k, f77_integer &n, f77_integer *qsiz, float *d__, float *q, f77_integer &ldq, f77_integer *indxq, float *rho, f77_integer *cutpnt, float *z__, float *dlamda, float *q2, f77_integer &ldq2, float *w, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, float *givnum, f77_integer *indxp, f77_integer *indx, f77_integer *info);
//  static inline void laed9(f77_integer &k, f77_integer *kstart, f77_integer *kstop, f77_integer &n, float *d__, float *q, f77_integer &ldq, float *rho, float *dlamda, float *w, float *s, f77_integer &lds, f77_integer *info);
//  static inline void laeda(f77_integer &n, f77_integer *tlvls, f77_integer *curlvl, f77_integer *curpbm, f77_integer *prmptr, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, float *givnum, float *q, f77_integer *qptr, float *z__, float *ztemp, f77_integer *info);
//  static inline void laein(f77_logical *rightv, f77_logical *noinit, f77_integer &n, float *h__, f77_integer &ldh, float *wr, float *wi, float *vr, float *vi, float *b, f77_integer &ldb, float *work, float *eps3, float *smlnum, float *bignum, f77_integer *info);
//  static inline void laev2(float *a, float *b, float *c__, float *rt1, float *rt2, float *cs1, float *sn1);
//  static inline void laexc(f77_logical &wantq, f77_integer &n, float *t, f77_integer &ldt, float *q, f77_integer &ldq, f77_integer *j1, f77_integer *n1, f77_integer *n2, float *work, f77_integer *info);
//  static inline void lag2(float *a, f77_integer &lda, float *b, f77_integer &ldb, float *safmin, float *scale1, float *scale2, float *wr1, float *wr2, float *wi);
//  static inline void lags2(f77_logical *upper, float *a1, float *a2, float *a3, float *b1, float *b2, float *b3, float *csu, float *snu, float *csv, float *snv, float *csq, float *snq);
//  static inline void lagtf(f77_integer &n, float *a, float *lambda, float *b, float *c__, float *tol, float *d__, f77_integer *in, f77_integer *info);
//  static inline void lagtm(const char *trans, f77_integer &n, f77_integer &nrhs, float *alpha, float *dl, float *d__, float *du, float *x, f77_integer &ldx, float *beta, float *b, f77_integer &ldb);
//  static inline void lagts(f77_integer *job, f77_integer &n, float *a, float *b, float *c__, float *d__, f77_integer *in, float *y, float *tol, f77_integer *info);
//  static inline void lagv2(float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *csl, float *snl, float *csr, float *snr);
//  static inline void lahqr(f77_logical &wantt, f77_logical &wantz, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *h__, f77_integer &ldh, float *wr, float *wi, f77_integer *iloz, f77_integer *ihiz, float *z__, f77_integer &ldz, f77_integer *info);
//  static inline void lahrd(f77_integer &n, f77_integer &k, f77_integer *nb, float *a, f77_integer &lda, float *tau, float *t, f77_integer &ldt, float *y, f77_integer &ldy);
//  static inline void laic1(f77_integer *job, f77_integer *j, float *x, float *sest, float *w, float *gamma, float *sestpr, float *s, float *c__);
//  static inline void laln2(f77_logical *ltrans, f77_integer *na, f77_integer *nw, float *smin, float *ca, float *a, f77_integer &lda, float *d1, float *d2, float *b, f77_integer &ldb, float *wr, float *wi, float *x, f77_integer &ldx, float *scale, float *xnorm, f77_integer *info);
//  static inline void lals0(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer &nrhs, float *b, f77_integer &ldb, float *bx, f77_integer &ldbx, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer &ldgcol, float *givnum, f77_integer &ldgnum, float *poles, float *difl, float *difr, float *z__, f77_integer &k, float *c__, float *s, float *work, f77_integer *info);
//  static inline void lalsa(f77_integer *icompq, f77_integer *smlsiz, f77_integer &n, f77_integer &nrhs, float *b, f77_integer &ldb, float *bx, f77_integer &ldbx, float *u, f77_integer &ldu, float *vt, f77_integer &k, float *difl, float *difr, float *z__, float *poles, f77_integer *givptr, f77_integer *givcol, f77_integer &ldgcol, f77_integer *perm, float *givnum, float *c__, float *s, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lalsd(const char *uplo, f77_integer *smlsiz, f77_integer &n, f77_integer &nrhs, float *d__, float *e, float *b, f77_integer &ldb, float *rcond, f77_integer *rank, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lamc1(f77_integer *beta, f77_integer *t, f77_logical *rnd, f77_logical *ieee1);
//  static inline void lamc2(f77_integer *beta, f77_integer *t, f77_logical *rnd, float *eps, f77_integer *emin, float *rmin, f77_integer *emax, float *rmax);
//  static inline void lamc4(f77_integer *emin, float *start, f77_integer *base);
//  static inline void lamc5(f77_integer *beta, f77_integer &p, f77_integer *emin, f77_logical *ieee, f77_integer *emax, float *rmax);
//  static inline void lamrg(f77_integer *n1, f77_integer *n2, float *a, f77_integer *dtrd1, f77_integer *dtrd2, f77_integer *index);
//  static inline void lanv2(float *a, float *b, float *c__, float *d__, float *rt1r, float *rt1i, float *rt2r, float *rt2i, float *cs, float *sn);
//  static inline void lapll(f77_integer &n, float *x, f77_integer &incx, float *y, f77_integer &incy, float *ssmin);
//  static inline void lapmt(f77_logical *forwrd, f77_integer &m, f77_integer &n, float *x, f77_integer &ldx, f77_integer &k);
//  static inline void laqgb(f77_integer &m, f77_integer &n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer &ldab, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, const char *equed);
//  static inline void laqge(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, const char *equed);
//  static inline void laqp2(f77_integer &m, f77_integer &n, f77_integer *offset, float *a, f77_integer &lda, f77_integer *jpvt, float *tau, float *vn1, float *vn2, float *work);
//  static inline void laqps(f77_integer &m, f77_integer &n, f77_integer *offset, f77_integer *nb, f77_integer *kb, float *a, f77_integer &lda, f77_integer *jpvt, float *tau, float *vn1, float *vn2, float *auxv, float *f, f77_integer &ldf);
//  static inline void laqsb(const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqsp(const char *uplo, f77_integer &n, float *ap, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqsy(const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqtr(f77_logical *ltran, f77_logical *lf77_real, f77_integer &n, float *t, f77_integer &ldt, float *b, float *w, float *scale, float *x, float *work, f77_integer *info);
//  static inline void lar1v(f77_integer &n, f77_integer *b1, f77_integer *bn, float *sigma, float *d__, float *l, float *ld, float *lld, float *gersch, float *z__, float *ztz, float *mingma, f77_integer *r__, f77_integer *isuppz, float *work);
//  static inline void lar2v(f77_integer &n, float *x, float *y, float *z__, f77_integer &incx, float *c__, float *s, f77_integer &incc);
//  static inline void larf(const char *side, f77_integer &m, f77_integer &n, float *v, f77_integer &incv, float *tau, float *c__, f77_integer &ldc, float *work);
//  static inline void larfb(const char *side, const char *trans, const char *direct, const char *storev, f77_integer &m, f77_integer &n, f77_integer &k, float *v, f77_integer &ldv, float *t, f77_integer &ldt, float *c__, f77_integer &ldc, float *work, f77_integer &ldwork);
//  static inline void larfg(f77_integer &n, float *alpha, float *x, f77_integer &incx, float *tau);
//  static inline void larft(const char *direct, const char *storev, f77_integer &n, f77_integer &k, float *v, f77_integer &ldv, float *tau, float *t, f77_integer &ldt);
//  static inline void larfx(const char *side, f77_integer &m, f77_integer &n, float *v, float *tau, float *c__, f77_integer &ldc, float *work);
//  static inline void largv(f77_integer &n, float *x, f77_integer &incx, float *y, f77_integer &incy, float *c__, f77_integer &incc);
//  static inline void larnv(f77_integer *idist, f77_integer *iseed, f77_integer &n, float *x);
//  static inline void larrb(f77_integer &n, float *d__, float *l, float *ld, float *lld, f77_integer *ifirst, f77_integer *ilast, float *sigma, float *reltol, float *w, float *wgap, float *werr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void larre(f77_integer &n, float *d__, float *e, float *tol, f77_integer *nsplit, f77_integer *isplit, f77_integer *m_out, float *w, float *woff, float *gersch, float *work, f77_integer *info);
//  static inline void larrf(f77_integer &n, float *d__, float *l, float *ld, float *lld, f77_integer *ifirst, f77_integer *ilast, float *w, float *dplus, float *lplus, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void larrv(f77_integer &n, float *d__, float *l, f77_integer *isplit, f77_integer &m, float *w, f77_integer *iblock, float *gersch, float *tol, float *z__, f77_integer &ldz, f77_integer *isuppz, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lartg(float *f, float *g, float *cs, float *sn, float *r__);
//  static inline void lartv(f77_integer &n, float *x, f77_integer &incx, float *y, f77_integer &incy, float *c__, float *s, f77_integer &incc);
//  static inline void laruv(f77_integer *iseed, f77_integer &n, float *x);
//  static inline void larz(const char *side, f77_integer &m, f77_integer &n, f77_integer *l, float *v, f77_integer &incv, float *tau, float *c__, f77_integer &ldc, float *work);
//  static inline void larzb(const char *side, const char *trans, const char *direct, const char *storev, f77_integer &m, f77_integer &n, f77_integer &k, f77_integer *l, float *v, f77_integer &ldv, float *t, f77_integer &ldt, float *c__, f77_integer &ldc, float *work, f77_integer &ldwork);
//  static inline void larzt(const char *direct, const char *storev, f77_integer &n, f77_integer &k, float *v, f77_integer &ldv, float *tau, float *t, f77_integer &ldt);
//  static inline void las2(float *f, float *g, float *h__, float *ssmin, float *ssmax);
//  static inline void lascl(const char *type__, f77_integer *kl, f77_integer *ku, float *cfrom, float *cto, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
//  static inline void lasd0(f77_integer &n, f77_integer *sqre, float *d__, float *e, float *u, f77_integer &ldu, float *vt, f77_integer &ldvt, f77_integer *smlsiz, f77_integer *iwork, float *work, f77_integer *info);
//  static inline void lasd1(f77_integer *nl, f77_integer *nr, f77_integer *sqre, float *d__, float *alpha, float *beta, float *u, f77_integer &ldu, float *vt, f77_integer &ldvt, f77_integer *idxq, f77_integer *iwork, float *work, f77_integer *info);
//  static inline void lasd2(f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer &k, float *d__, float *z__, float *alpha, float *beta, float *u, f77_integer &ldu, float *vt, f77_integer &ldvt, float *dsigma, float *u2, f77_integer &ldu2, float *vt2, f77_integer &ldvt2, f77_integer *idxp, f77_integer *idx, f77_integer *idxc, f77_integer *idxq, f77_integer *coltyp, f77_integer *info);
//  static inline void lasd3(f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer &k, float *d__, float *q, f77_integer &ldq, float *dsigma, float *u, f77_integer &ldu, float *u2, f77_integer &ldu2, float *vt, f77_integer &ldvt, float *vt2, f77_integer &ldvt2, f77_integer *idxc, f77_integer *ctot, float *z__, f77_integer *info);
//  static inline void lasd4(f77_integer &n, f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *sigma, float *work, f77_integer *info);
//  static inline void lasd5(f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *dsigma, float *work);
//  static inline void lasd6(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, float *d__, float *vf, float *vl, float *alpha, float *beta, f77_integer *idxq, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer &ldgcol, float *givnum, f77_integer &ldgnum, float *poles, float *difl, float *difr, float *z__, f77_integer &k, float *c__, float *s, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lasd7(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer &k, float *d__, float *z__, float *zw, float *vf, float *vfw, float *vl, float *vlw, float *alpha, float *beta, float *dsigma, f77_integer *idx, f77_integer *idxp, f77_integer *idxq, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer &ldgcol, float *givnum, f77_integer &ldgnum, float *c__, float *s, f77_integer *info);
//  static inline void lasd8(f77_integer *icompq, f77_integer &k, float *d__, float *z__, float *vf, float *vl, float *difl, float *difr, f77_integer &lddifr, float *dsigma, float *work, f77_integer *info);
//  static inline void lasd9(f77_integer *icompq, f77_integer &ldu, f77_integer &k, float *d__, float *z__, float *vf, float *vl, float *difl, float *difr, float *dsigma, float *work, f77_integer *info);
//  static inline void lasda(f77_integer *icompq, f77_integer *smlsiz, f77_integer &n, f77_integer *sqre, float *d__, float *e, float *u, f77_integer &ldu, float *vt, f77_integer &k, float *difl, float *difr, float *z__, float *poles, f77_integer *givptr, f77_integer *givcol, f77_integer &ldgcol, f77_integer *perm, float *givnum, float *c__, float *s, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lasdq(const char *uplo, f77_integer *sqre, f77_integer &n, f77_integer &ncvt, f77_integer &nru, f77_integer &ncc, float *d__, float *e, float *vt, f77_integer &ldvt, float *u, f77_integer &ldu, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void lasdt(f77_integer &n, f77_integer *lvl, f77_integer *nd, f77_integer *inode, f77_integer *ndiml, f77_integer *ndimr, f77_integer *msub);
//  static inline void laset(const char *uplo, f77_integer &m, f77_integer &n, float *alpha, float *beta, float *a, f77_integer &lda);
//  static inline void lasq1(f77_integer &n, float *d__, float *e, float *work, f77_integer *info);
//  static inline void lasq2(f77_integer &n, float *z__, f77_integer *info);
//  static inline void lasq3(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, float *dmin__, float *sigma, float *desig, float *qmax, f77_integer *nfail, f77_integer *iter, f77_integer *ndiv, f77_logical *ieee);
//  static inline void lasq4(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, f77_integer *n0in, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dn1, float *dn2, float *tau, f77_integer *ttype);
//  static inline void lasq5(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, float *tau, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dnm1, float *dnm2, f77_logical *ieee);
//  static inline void lasq6(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dnm1, float *dnm2);
//  static inline void lasr(const char *side, const char *pivot, const char *direct, f77_integer &m, f77_integer &n, float *c__, float *s, float *a, f77_integer &lda);
//  static inline void lasrt(const char *id, f77_integer &n, float *d__, f77_integer *info);
//  static inline void lassq(f77_integer &n, float *x, f77_integer &incx, float *scale, float *sumsq);
//  static inline void lasv2(float *f, float *g, float *h__, float *ssmin, float *ssmax, float *snr, float *csr, float *snl, float *csl);
//  static inline void laswp(f77_integer &n, float *a, f77_integer &lda, f77_integer *k1, f77_integer *k2, f77_integer *ipiv, f77_integer &incx);
//  static inline void lasy2(f77_logical *ltranl, f77_logical *ltranr, f77_integer *isgn, f77_integer *n1, f77_integer *n2, float *tl, f77_integer &ldtl, float *tr, f77_integer &ldtr, float *b, f77_integer &ldb, float *scale, float *x, f77_integer &ldx, float *xnorm, f77_integer *info);
//  static inline void lasyf(const char *uplo, f77_integer &n, f77_integer *nb, f77_integer *kb, float *a, f77_integer &lda, f77_integer *ipiv, float *w, f77_integer &ldw, f77_integer *info);
//  static inline void latbs(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *x, float *scale, float *cnorm, f77_integer *info);
//  static inline void latdf(f77_integer *ijob, f77_integer &n, float *z__, f77_integer &ldz, float *rhs, float *rdsum, float *rdscal, f77_integer *ipiv, f77_integer *jpiv);
//  static inline void latps(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer &n, float *ap, float *x, float *scale, float *cnorm, f77_integer *info);
//  static inline void latrd(const char *uplo, f77_integer &n, f77_integer *nb, float *a, f77_integer &lda, float *e, float *tau, float *w, f77_integer &ldw);
//  static inline void latrs(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer &n, float *a, f77_integer &lda, float *x, float *scale, float *cnorm, f77_integer *info);
//  static inline void latrz(f77_integer &m, f77_integer &n, f77_integer *l, float *a, f77_integer &lda, float *tau, float *work);
//  static inline void latzm(const char *side, f77_integer &m, f77_integer &n, float *v, f77_integer &incv, float *tau, float *c1, float *c2, f77_integer &ldc, float *work);
//  static inline void lauu2(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
//  static inline void lauum(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
//  static inline void opgtr(const char *uplo, f77_integer &n, float *ap, float *tau, float *q, f77_integer &ldq, float *work, f77_integer *info);
//  static inline void opmtr(const char *side, const char *uplo, const char *trans, f77_integer &m, f77_integer &n, float *ap, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void org2l(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void org2r(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void orgbr(const char *vect, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void orghr(f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void orgl2(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void orglq(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void orgql(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
  static inline void orgqr(f77_integer m, f77_integer n, f77_integer k, float *a, f77_integer lda, float *tau, float *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(sorgqr)(m, n, k, a, lda, tau, work, lwork, info);

  }
//  static inline void orgr2(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer *info);
//  static inline void orgrq(f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void orgtr(const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void orm2l(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void orm2r(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void ormbr(const char *vect, const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ormhr(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void orml2(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void ormlq(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ormql(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ormqr(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ormr2(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void ormr3(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, f77_integer *l, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer *info);
//  static inline void ormrq(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ormrz(const char *side, const char *trans, f77_integer &m, f77_integer &n, f77_integer &k, f77_integer *l, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void ormtr(const char *side, const char *uplo, const char *trans, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *c__, f77_integer &ldc, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void pbcon(const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbequ(const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *s, float *scond, float *amax, f77_integer *info);
//  static inline void pbrfs(const char *uplo, f77_integer &n, f77_integer *kd, f77_integer &nrhs, float *ab, f77_integer &ldab, float *afb, f77_integer &ldafb, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbstf(const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, f77_integer *info);
//  static inline void pbsv(const char *uplo, f77_integer &n, f77_integer *kd, f77_integer &nrhs, float *ab, f77_integer &ldab, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void pbsvx(const char *fact, const char *uplo, f77_integer &n, f77_integer *kd, f77_integer &nrhs, float *ab, f77_integer &ldab, float *afb, f77_integer &ldafb, const char *equed, float *s, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbtf2(const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, f77_integer *info);
//  static inline void pbtrf(const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, f77_integer *info);
//  static inline void pbtrs(const char *uplo, f77_integer &n, f77_integer *kd, f77_integer &nrhs, float *ab, f77_integer &ldab, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void pocon(const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void poequ(f77_integer &n, float *a, f77_integer &lda, float *s, float *scond, float *amax, f77_integer *info);
//  static inline void porfs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *af, f77_integer &ldaf, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void posv(const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void posvx(const char *fact, const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *af, f77_integer &ldaf, const char *equed, float *s, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void potf2(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
  static inline void potrf(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *info) {
    potrf(uplo, n, a, lda, info);
  }
//  static inline void potri(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
//  static inline void potrs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void ppcon(const char *uplo, f77_integer &n, float *ap, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void ppequ(const char *uplo, f77_integer &n, float *ap, float *s, float *scond, float *amax, f77_integer *info);
//  static inline void pprfs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, float *afp, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void ppsv(const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void ppsvx(const char *fact, const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, float *afp, const char *equed, float *s, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pptrf(const char *uplo, f77_integer &n, float *ap, f77_integer *info);
//  static inline void pptri(const char *uplo, f77_integer &n, float *ap, f77_integer *info);
//  static inline void pptrs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void ptcon(f77_integer &n, float *d__, float *e, float *anorm, float *rcond, float *work, f77_integer *info);
//  static inline void pteqr(const char *compz, f77_integer &n, float *d__, float *e, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void ptrfs(f77_integer &n, f77_integer &nrhs, float *d__, float *e, float *df, float *ef, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *info);
//  static inline void ptsv(f77_integer &n, f77_integer &nrhs, float *d__, float *e, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void ptsvx(const char *fact, f77_integer &n, f77_integer &nrhs, float *d__, float *e, float *df, float *ef, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *info);
//  static inline void pttrf(f77_integer &n, float *d__, float *e, f77_integer *info);
//  static inline void pttrs(f77_integer &n, f77_integer &nrhs, float *d__, float *e, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void ptts2(f77_integer &n, f77_integer &nrhs, float *d__, float *e, float *b, f77_integer &ldb);
//  static inline void rscl(f77_integer &n, float *sa, float *sx, f77_integer &incx);
//  static inline void sbev(const char *jobz, const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void sbevd(const char *jobz, const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *w, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void sbevx(const char *jobz, const char *range, const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *q, f77_integer &ldq, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sbgst(const char *vect, const char *uplo, f77_integer &n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer &ldab, float *bb, f77_integer &ldbb, float *x, f77_integer &ldx, float *work, f77_integer *info);
//  static inline void sbgv(const char *jobz, const char *uplo, f77_integer &n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer &ldab, float *bb, f77_integer &ldbb, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void sbgvd(const char *jobz, const char *uplo, f77_integer &n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer &ldab, float *bb, f77_integer &ldbb, float *w, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void sbgvx(const char *jobz, const char *range, const char *uplo, f77_integer &n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer &ldab, float *bb, f77_integer &ldbb, float *q, f77_integer &ldq, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sbtrd(const char *vect, const char *uplo, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *d__, float *e, float *q, f77_integer &ldq, float *work, f77_integer *info);
//  static inline void spcon(const char *uplo, f77_integer &n, float *ap, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void spev(const char *jobz, const char *uplo, f77_integer &n, float *ap, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void spevd(const char *jobz, const char *uplo, f77_integer &n, float *ap, float *w, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void spevx(const char *jobz, const char *range, const char *uplo, f77_integer &n, float *ap, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void spgst(f77_integer *itype, const char *uplo, f77_integer &n, float *ap, float *bp, f77_integer *info);
//  static inline void spgv(f77_integer *itype, const char *jobz, const char *uplo, f77_integer &n, float *ap, float *bp, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void spgvd(f77_integer *itype, const char *jobz, const char *uplo, f77_integer &n, float *ap, float *bp, float *w, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void spgvx(f77_integer *itype, const char *jobz, const char *range, const char *uplo, f77_integer &n, float *ap, float *bp, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sprfs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, float *afp, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void spsv(const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void spsvx(const char *fact, const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, float *afp, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void sptrd(const char *uplo, f77_integer &n, float *ap, float *d__, float *e, float *tau, f77_integer *info);
//  static inline void sptrf(const char *uplo, f77_integer &n, float *ap, f77_integer *ipiv, f77_integer *info);
//  static inline void sptri(const char *uplo, f77_integer &n, float *ap, f77_integer *ipiv, float *work, f77_integer *info);
//  static inline void sptrs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *ap, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void stebz(const char *range, const char *order, f77_integer &n, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, float *d__, float *e, f77_integer *m_out, f77_integer *nsplit, float *w, f77_integer *iblock, f77_integer *isplit, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void stedc(const char *compz, f77_integer &n, float *d__, float *e, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void stegr(const char *jobz, const char *range, f77_integer &n, float *d__, float *e, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, f77_integer *isuppz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void stein(f77_integer &n, float *d__, float *e, f77_integer &m, float *w, f77_integer *iblock, f77_integer *isplit, float *z__, f77_integer &ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void steqr(const char *compz, f77_integer &n, float *d__, float *e, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void sterf(f77_integer &n, float *d__, float *e, f77_integer *info);
//  static inline void stev(const char *jobz, f77_integer &n, float *d__, float *e, float *z__, f77_integer &ldz, float *work, f77_integer *info);
//  static inline void stevd(const char *jobz, f77_integer &n, float *d__, float *e, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void stevr(const char *jobz, const char *range, f77_integer &n, float *d__, float *e, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, f77_integer *isuppz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void stevx(const char *jobz, const char *range, f77_integer &n, float *d__, float *e, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sycon(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void syev(const char *jobz, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *w, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void syevd(const char *jobz, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *w, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void syevr(const char *jobz, const char *range, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, f77_integer *isuppz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void syevx(const char *jobz, const char *range, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sygs2(f77_integer *itype, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void sygst(f77_integer *itype, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *info);
  static inline void sygv(f77_integer *itype, const char *jobz, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *w, float *work, f77_integer &lwork, f77_integer *info) {
    F77_FUNC(ssygv)(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);
  }
//  static inline void sygvd(f77_integer *itype, const char *jobz, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *w, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void sygvx(f77_integer *itype, const char *jobz, const char *range, const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer &ldz, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void syrfs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *af, f77_integer &ldaf, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void sysv(const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, f77_integer *ipiv, float *b, f77_integer &ldb, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void sysvx(const char *fact, const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *af, f77_integer &ldaf, f77_integer *ipiv, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *info);
//  static inline void sytd2(const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *d__, float *e, float *tau, f77_integer *info);
//  static inline void sytf2(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, f77_integer *info);
//  static inline void sytrd(const char *uplo, f77_integer &n, float *a, f77_integer &lda, float *d__, float *e, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void sytrf(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void sytri(const char *uplo, f77_integer &n, float *a, f77_integer &lda, f77_integer *ipiv, float *work, f77_integer *info);
//  static inline void sytrs(const char *uplo, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, f77_integer *ipiv, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void tbcon(const char *norm, const char *uplo, const char *diag, f77_integer &n, f77_integer *kd, float *ab, f77_integer &ldab, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tbrfs(const char *uplo, const char *trans, const char *diag, f77_integer &n, f77_integer *kd, f77_integer &nrhs, float *ab, f77_integer &ldab, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tbtrs(const char *uplo, const char *trans, const char *diag, f77_integer &n, f77_integer *kd, f77_integer &nrhs, float *ab, f77_integer &ldab, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void tgevc(const char *side, const char *howmny, f77_logical *select, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, f77_integer *mm, f77_integer *m_out, float *work, f77_integer *info);
//  static inline void tgex2(f77_logical &wantq, f77_logical &wantz, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *q, f77_integer &ldq, float *z__, f77_integer &ldz, f77_integer *j1, f77_integer *n1, f77_integer *n2, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void tgexc(f77_logical &wantq, f77_logical &wantz, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *q, f77_integer &ldq, float *z__, f77_integer &ldz, f77_integer *ifst, f77_integer *ilst, float *work, f77_integer &lwork, f77_integer *info);
//  static inline void tgsen(f77_integer *ijob, f77_logical &wantq, f77_logical &wantz, f77_logical *select, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *alphar, float *alphai, float *beta, float *q, f77_integer &ldq, float *z__, f77_integer &ldz, f77_integer *m_out, float *pl, float *pr, float *dif, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void tgsja(const char *jobu, const char *jobv, const char *jobq, f77_integer &m, f77_integer &p, f77_integer &n, f77_integer &k, f77_integer *l, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *tola, float *tolb, float *alpha, float *beta, float *u, f77_integer &ldu, float *v, f77_integer &ldv, float *q, f77_integer &ldq, float *work, f77_integer *ncycle, f77_integer *info);
//  static inline void tgsna(const char *job, const char *howmny, f77_logical *select, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, float *s, float *dif, f77_integer *mm, f77_integer *m_out, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *info);
//  static inline void tgsy2(const char *trans, f77_integer *ijob, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *c__, f77_integer &ldc, float *d__, f77_integer &ldd, float *e, f77_integer &lde, float *f, f77_integer &ldf, float *scale, float *rdsum, float *rdscal, f77_integer *iwork, f77_integer *pq, f77_integer *info);
//  static inline void tgsyl(const char *trans, f77_integer *ijob, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *c__, f77_integer &ldc, float *d__, f77_integer &ldd, float *e, f77_integer &lde, float *f, f77_integer &ldf, float *scale, float *dif, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer *info);
//  static inline void tpcon(const char *norm, const char *uplo, const char *diag, f77_integer &n, float *ap, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tprfs(const char *uplo, const char *trans, const char *diag, f77_integer &n, f77_integer &nrhs, float *ap, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tptri(const char *uplo, const char *diag, f77_integer &n, float *ap, f77_integer *info);
//  static inline void tptrs(const char *uplo, const char *trans, const char *diag, f77_integer &n, f77_integer &nrhs, float *ap, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void trcon(const char *norm, const char *uplo, const char *diag, f77_integer &n, float *a, f77_integer &lda, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void trevc(const char *side, const char *howmny, f77_logical *select, f77_integer &n, float *t, f77_integer &ldt, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, f77_integer *mm, f77_integer *m_out, float *work, f77_integer *info);
//  static inline void trexc(const char *compq, f77_integer &n, float *t, f77_integer &ldt, float *q, f77_integer &ldq, f77_integer *ifst, f77_integer *ilst, float *work, f77_integer *info);
//  static inline void trrfs(const char *uplo, const char *trans, const char *diag, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *x, f77_integer &ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void trsen(const char *job, const char *compq, f77_logical *select, f77_integer &n, float *t, f77_integer &ldt, float *q, f77_integer &ldq, float *wr, float *wi, f77_integer *m_out, float *s, float *sep, float *work, f77_integer &lwork, f77_integer *iwork, f77_integer &liwork, f77_integer *info);
//  static inline void trsna(const char *job, const char *howmny, f77_logical *select, f77_integer &n, float *t, f77_integer &ldt, float *vl, f77_integer &ldvl, float *vr, f77_integer &ldvr, float *s, float *sep, f77_integer *mm, f77_integer *m_out, float *work, f77_integer &ldwork, f77_integer *iwork, f77_integer *info);
//  static inline void trsyl(const char *trana, const char *tranb, f77_integer *isgn, f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *b, f77_integer &ldb, float *c__, f77_integer &ldc, float *scale, f77_integer *info);
//  static inline void trti2(const char *uplo, const char *diag, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
//  static inline void trtri(const char *uplo, const char *diag, f77_integer &n, float *a, f77_integer &lda, f77_integer *info);
//  static inline void trtrs(const char *uplo, const char *trans, const char *diag, f77_integer &n, f77_integer &nrhs, float *a, f77_integer &lda, float *b, f77_integer &ldb, f77_integer *info);
//  static inline void tzrqf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, f77_integer *info);
//  static inline void tzrzf(f77_integer &m, f77_integer &n, float *a, f77_integer &lda, float *tau, float *work, f77_integer &lwork, f77_integer *info);
//  f77_integer icmax1(f77_integer &n, f77_complex *cx, f77_integer &incx);
//  f77_integer ieeeck(f77_integer &ispec, f77_real *zero, f77_real *one);
//  f77_integer ilaenv(f77_integer &ispec, const char *name__, const char *opts, f77_integer *n1, f77_integer *n2, f77_integer *n3, f77_integer *n4, f77_str_len name_len, f77_str_len opts_len);
//  f77_integer izmax1(f77_integer &n, floatcomplex *cx, f77_integer &incx);

}; //CppLapack

static CppLapack<float> cpplapack_float;

}; //la name space

#endif
