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


  static inline void bdsdc(const char *uplo, const char *compq, f77_integer CONST_REF n, float *d__, float *e, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF ldvt, float *q, f77_integer *iq, float *work, f77_integer *iwork, f77_integer *info) {
     F77_FUNC(sbdsdc)(uplo, compq, n, d__, e, u, ldu, vt, ldvt, q, iq, work, iwork, info); 
  }
  static inline void bdsqr(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF ncvt, f77_integer CONST_REF nru, f77_integer CONST_REF ncc, float *d__, float *e, float *vt, f77_integer CONST_REF ldvt, float *u, f77_integer CONST_REF ldu, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info) {
     F77_FUNC(sbdsqr)(uplo, n, ncvt, nru, ncc, d__, e, vt, ldvt, u, ldu, c__, ldc, work, info);
  }
  static inline void disna(const char *job, f77_integer CONST_REF m, f77_integer CONST_REF n, float *d__, float *sep, f77_integer *info) {
     F77_FUNC(sdisna)(job, m, n, d__, sep, info);
  }
  static inline void gbbrd(const char *vect, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF ncc, f77_integer *kl, f77_integer *ku, float *ab, f77_integer CONST_REF ldab, float *d__, float *e, float *q, f77_integer CONST_REF ldq, float *pt, f77_integer CONST_REF ldpt, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info) {
     F77_FUNC(sgbbrd)(vect, m, n, ncc, kl, ku, ab, ldab, d__, e, q, ldq, pt, ldpt, c__, ldc, work, info);
  }
  static inline void gbcon(const char *norm, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info) {
     F77_FUNC(sgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork, info);
  }
  static inline void gbequ(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer CONST_REF ldab, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, f77_integer *info) {
     F77_FUNC(sgbequ)(m, n, kl, ku, ab, ldab, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gbrfs(const char *trans, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *afb, f77_integer CONST_REF ldafb, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer*iwork, f77_integer *info) {
    F77_FUNC(sgbrfs)(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  }
  static inline void gbsv(f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info) {
    F77_FUNC(sgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
  }
  static inline void gbsvx(const char *fact, const char *trans, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *afb, f77_integer CONST_REF ldafb, f77_integer *ipiv, const char *equed, float *r__, float *c__, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r__, c__, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
  }
  static inline void gbtf2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(sgbtf2)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrf(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(sgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrs(const char *trans, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info) {
    F77_FUNC(sgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);

  }
  static inline void gebak(const char *job, const char *side, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *scale, f77_integer CONST_REF m, float *v, f77_integer CONST_REF ldv, f77_integer *info) {
    F77_FUNC(sgebak)(job, side, n, ilo, ihi, scale, m, v, ldv, info);
  }
  static inline void gebal(const char *job, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ilo, f77_integer *ihi, float *scale, f77_integer *info) {
    F77_FUNC(sgebal)(job, n, a, lda, ilo, ihi, scale, info);

  }
  static inline void gebd2(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *d__, float *e, float *tauq, float *taup, float *work, f77_integer *info) {
    F77_FUNC(sgebd2)(m, n, a, lda, d__, e, tauq, taup, work, info);
  }
  static inline void gebrd(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *d__, float *e, float *tauq, float *taup, float *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(sgebrd)(m, n, a, lda, d__, e, tauq, taup, work, lwork, info);
  }
  static inline void gecon(const char *norm, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
  }
  static inline void geequ(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, f77_integer *info) {
    F77_FUNC(sgeequ)(m, n, a, lda, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gees(const char *jobvs, const char *sort, f77_logical_func select, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *sdim, float *wr, float *wi, float *vs, f77_integer CONST_REF ldvs, float *work, f77_integer CONST_REF lwork, unsigned int *bwork, f77_integer *info) {
    F77_FUNC(sgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
  }
  static inline void geesx(const char *jobvs, const char *sort, f77_logical_func select, const char *sense, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *sdim, float *wr, float *wi, float *vs, f77_integer CONST_REF ldvs, float *rconde, float *rcondv, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, unsigned int *bwork, f77_integer *info) {
    F77_FUNC(sgeesx)(jobvs, sort, select, sense, n, a, lda, sdim, wr, wi, vs, ldvs, rconde, rcondv, work, lwork, iwork, liwork, bwork, info);
  }
  static inline void geev(const char *jobvl, const char *jobvr, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *wr, float *wi, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, float *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(sgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
  }
  static inline void geevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *wr, float *wi, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, f77_integer *ilo, f77_integer *ihi, float *scale, float *abnrm, float *rconde, float *rcondv, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work,lwork, iwork, info);
  }
  static inline void gegs(const char *jobvsl, const char *jobvsr, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *vsl, f77_integer CONST_REF ldvsl, float *vsr, f77_integer CONST_REF ldvsr, float *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(sgegs)(jobvsl, jobvsr, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, info);
  }
  static inline void gegv(const char *jobvl, const char *jobvr, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, float *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(sgegv)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
  }
//  static inline void gehd2(f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void gehrd(f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gelq2(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void gelqf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gels(const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gelsd(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *s, float *rcond, f77_integer *rank, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void gelss(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *s, float *rcond, f77_integer *rank, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gelsx(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *jpvt, float *rcond, f77_integer *rank, float *work, f77_integer *info);
//  static inline void gelsy(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *jpvt, float *rcond, f77_integer *rank, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void geql2(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void geqlf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
// static inline void geqp3(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *jpvt, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void geqpf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *jpvt, float *tau, float *work, f77_integer *info);
//  static inline void geqr2(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
  static inline void geqrf(f77_integer m, f77_integer n, float *a, f77_integer lda, float *tau, float *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(sgeqrf)(m, n, a, lda, tau, work, lwork, info);
  }
  static inline void gerfs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgerfs)(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  
  }
//  static inline void gerq2(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void gerqf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gesc2(f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *rhs, f77_integer *ipiv, f77_integer *jpiv, float *scale);
  static inline void gesdd(const char *jobz, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *s, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF ldvt, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(sgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static inline void gesv(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info) {
    F77_FUNC(sgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
  }
//  static inline void gesvd(const char *jobu, const char *jobvt, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *s, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF ldvt, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gesvx(const char *fact, const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, const char *equed, float *r__, float *c__, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void getc2(f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *jpiv, f77_integer *info);
//  static inline void getf2(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *info);
  static inline void getrf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(sgetrf)(m, n, a, lda, ipiv, info); 
  }
  static inline void getri(f77_integer n, float *a, f77_integer lda, f77_integer *ipiv, float *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(sgetri)(n, a, lda, ipiv, work, lwork, info);
  }
//  static inline void getrs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ggbak(const char *job, const char *side, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *lscale, float *rscale, f77_integer CONST_REF m, float *v, f77_integer CONST_REF ldv, f77_integer *info);
//  static inline void ggbal(const char *job, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *ilo, f77_integer *ihi, float *lscale, float *rscale, float *work, f77_integer *info);
//  static inline void gges(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *sdim, float *alphar, float *alphai, float *beta, float *vsl, f77_integer CONST_REF ldvsl, float *vsr, f77_integer CONST_REF ldvsr, float *work, f77_integer CONST_REF lwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggesx(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, const char *sense, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *sdim, float *alphar, float *alphai, float *beta, float *vsl, f77_integer CONST_REF ldvsl, float *vsr, f77_integer CONST_REF ldvsr, float *rconde, float *rcondv, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggev(const char *jobvl, const char *jobvr, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, f77_integer *ilo, f77_integer *ihi, float *lscale, float *rscale, float *abnrm, float *bbnrm, float *rconde, float *rcondv, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggglm(f77_integer CONST_REF n, f77_integer CONST_REF m, f77_integer CONST_REF p, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *d__, float *x, float *y, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gghrd(const char *compq, const char *compz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *q, f77_integer CONST_REF ldq, float *z__, f77_integer CONST_REF ldz, f77_integer *info);
//  static inline void gglse(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF p, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *c__, float *d__, float *x, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggqrf(f77_integer CONST_REF n, f77_integer CONST_REF m, f77_integer CONST_REF p, float *a, f77_integer CONST_REF lda, float *taua, float *b, f77_integer CONST_REF ldb, float *taub, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggrqf(f77_integer CONST_REF m, f77_integer CONST_REF p, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *taua, float *b, f77_integer CONST_REF ldb, float *taub, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggsvd(const char *jobu, const char *jobv, const char *jobq, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF p, f77_integer CONST_REF k, f77_integer *l, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alpha, float *beta, float *u, f77_integer CONST_REF ldu, float *v, f77_integer CONST_REF ldv, float *q, f77_integer CONST_REF ldq, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void ggsvp(const char *jobu, const char *jobv, const char *jobq, f77_integer CONST_REF m, f77_integer CONST_REF p, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *tola, float *tolb, f77_integer CONST_REF k, f77_integer *l, float *u, f77_integer CONST_REF ldu, float *v, f77_integer CONST_REF ldv, float *q, f77_integer CONST_REF ldq, f77_integer *iwork, float *tau, float *work, f77_integer *info);
//  static inline void gtcon(const char *norm, f77_integer CONST_REF n, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void gtrfs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *dl, float *d__, float *du, float *dlf, float *df, float *duf, float *du2, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void gtsv(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *dl, float *d__, float *du, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void gtsvx(const char *fact, const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *dl, float *d__, float *du, float *dlf, float *df, float *duf, float *du2, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void gttrf(f77_integer CONST_REF n, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, f77_integer *info);
//  static inline void gttrs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void gtts2(f77_integer *itrans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *dl, float *d__, float *du, float *du2, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb);
//  static inline void hgeqz(const char *job, const char *compq, const char *compz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *q, f77_integer CONST_REF ldq, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void hsein(const char *side, const char *eigsrc, const char *initv, f77_logical *select, f77_integer CONST_REF n, float *h__, f77_integer CONST_REF ldh, float *wr, float *wi, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, f77_integer *mm, f77_integer *m_out, float *work, f77_integer *ifaill, f77_integer *ifailr, f77_integer *info);
//  static inline void hseqr(const char *job, const char *compz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *h__, f77_integer CONST_REF ldh, float *wr, float *wi, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void labad(float *small, float *large);
//  static inline void labrd(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *nb, float *a, f77_integer CONST_REF lda, float *d__, float *e, float *tauq, float *taup, float *x, f77_integer CONST_REF ldx, float *y, f77_integer CONST_REF ldy);
//  static inline void lacon(f77_integer CONST_REF n, float *v, float *x, f77_integer *isgn, float *est, f77_integer *kase);
//  static inline void lacpy(const char *uplo, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb);
//  static inline void ladiv(float *a, float *b, float *c__, float *d__, float *p, float *q);
//  static inline void lae2(float *a, float *b, float *c__, float *rt1, float *rt2);
//  static inline void laebz(f77_integer *ijob, f77_integer *nitmax, f77_integer CONST_REF n, f77_integer *mmax, f77_integer *minp, f77_integer *nbmin, float *abstol, float *reltol, float *pivmin, float *d__, float *e, float *e2, f77_integer *nval, float *ab, float *c__, f77_integer *mout, f77_integer *nab, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed0(f77_integer *icompq, f77_integer *qsiz, f77_integer CONST_REF n, float *d__, float *e, float *q, f77_integer CONST_REF ldq, float *qstore, f77_integer CONST_REF ldqs, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed1(f77_integer CONST_REF n, float *d__, float *q, f77_integer CONST_REF ldq, f77_integer *indxq, float *rho, f77_integer *cutpnt, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed2(f77_integer CONST_REF k, f77_integer CONST_REF n, f77_integer *n1, float *d__, float *q, f77_integer CONST_REF ldq, f77_integer *indxq, float *rho, float *z__, float *dlamda, float *w, float *q2, f77_integer *indx, f77_integer *indxc, f77_integer *indxp, f77_integer *coltyp, f77_integer *info);
//  static inline void laed3(f77_integer CONST_REF k, f77_integer CONST_REF n, f77_integer *n1, float *d__, float *q, f77_integer CONST_REF ldq, float *rho, float *dlamda, float *q2, f77_integer *indx, f77_integer *ctot, float *w, float *s, f77_integer *info);
//  static inline void laed4(f77_integer CONST_REF n, f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *dlam, f77_integer *info);
//  static inline void laed5(f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *dlam);
//  static inline void laed6(f77_integer *kniter, f77_logical *orgati, float *rho, float *d__, float *z__, float *finit, float *tau, f77_integer *info);
//  static inline void laed7(f77_integer *icompq, f77_integer CONST_REF n, f77_integer *qsiz, f77_integer *tlvls, f77_integer *curlvl, f77_integer *curpbm, float *d__, float *q, f77_integer CONST_REF ldq, f77_integer *indxq, float *rho, f77_integer *cutpnt, float *qstore, f77_integer *qptr, f77_integer *prmptr, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, float *givnum, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed8(f77_integer *icompq, f77_integer CONST_REF k, f77_integer CONST_REF n, f77_integer *qsiz, float *d__, float *q, f77_integer CONST_REF ldq, f77_integer *indxq, float *rho, f77_integer *cutpnt, float *z__, float *dlamda, float *q2, f77_integer CONST_REF ldq2, float *w, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, float *givnum, f77_integer *indxp, f77_integer *indx, f77_integer *info);
//  static inline void laed9(f77_integer CONST_REF k, f77_integer *kstart, f77_integer *kstop, f77_integer CONST_REF n, float *d__, float *q, f77_integer CONST_REF ldq, float *rho, float *dlamda, float *w, float *s, f77_integer CONST_REF lds, f77_integer *info);
//  static inline void laeda(f77_integer CONST_REF n, f77_integer *tlvls, f77_integer *curlvl, f77_integer *curpbm, f77_integer *prmptr, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, float *givnum, float *q, f77_integer *qptr, float *z__, float *ztemp, f77_integer *info);
//  static inline void laein(f77_logical *rightv, f77_logical *noinit, f77_integer CONST_REF n, float *h__, f77_integer CONST_REF ldh, float *wr, float *wi, float *vr, float *vi, float *b, f77_integer CONST_REF ldb, float *work, float *eps3, float *smlnum, float *bignum, f77_integer *info);
//  static inline void laev2(float *a, float *b, float *c__, float *rt1, float *rt2, float *cs1, float *sn1);
//  static inline void laexc(f77_logical CONST_REF wantq, f77_integer CONST_REF n, float *t, f77_integer CONST_REF ldt, float *q, f77_integer CONST_REF ldq, f77_integer *j1, f77_integer *n1, f77_integer *n2, float *work, f77_integer *info);
//  static inline void lag2(float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *safmin, float *scale1, float *scale2, float *wr1, float *wr2, float *wi);
//  static inline void lags2(f77_logical *upper, float *a1, float *a2, float *a3, float *b1, float *b2, float *b3, float *csu, float *snu, float *csv, float *snv, float *csq, float *snq);
//  static inline void lagtf(f77_integer CONST_REF n, float *a, float *lambda, float *b, float *c__, float *tol, float *d__, f77_integer *in, f77_integer *info);
//  static inline void lagtm(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *alpha, float *dl, float *d__, float *du, float *x, f77_integer CONST_REF ldx, float *beta, float *b, f77_integer CONST_REF ldb);
//  static inline void lagts(f77_integer *job, f77_integer CONST_REF n, float *a, float *b, float *c__, float *d__, f77_integer *in, float *y, float *tol, f77_integer *info);
//  static inline void lagv2(float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *csl, float *snl, float *csr, float *snr);
//  static inline void lahqr(f77_logical CONST_REF wantt, f77_logical CONST_REF wantz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *h__, f77_integer CONST_REF ldh, float *wr, float *wi, f77_integer *iloz, f77_integer *ihiz, float *z__, f77_integer CONST_REF ldz, f77_integer *info);
//  static inline void lahrd(f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *nb, float *a, f77_integer CONST_REF lda, float *tau, float *t, f77_integer CONST_REF ldt, float *y, f77_integer CONST_REF ldy);
//  static inline void laic1(f77_integer *job, f77_integer *j, float *x, float *sest, float *w, float *gamma, float *sestpr, float *s, float *c__);
//  static inline void laln2(f77_logical *ltrans, f77_integer *na, f77_integer *nw, float *smin, float *ca, float *a, f77_integer CONST_REF lda, float *d1, float *d2, float *b, f77_integer CONST_REF ldb, float *wr, float *wi, float *x, f77_integer CONST_REF ldx, float *scale, float *xnorm, f77_integer *info);
//  static inline void lals0(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF nrhs, float *b, f77_integer CONST_REF ldb, float *bx, f77_integer CONST_REF ldbx, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, float *givnum, f77_integer CONST_REF ldgnum, float *poles, float *difl, float *difr, float *z__, f77_integer CONST_REF k, float *c__, float *s, float *work, f77_integer *info);
//  static inline void lalsa(f77_integer *icompq, f77_integer *smlsiz, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *b, f77_integer CONST_REF ldb, float *bx, f77_integer CONST_REF ldbx, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF k, float *difl, float *difr, float *z__, float *poles, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, f77_integer *perm, float *givnum, float *c__, float *s, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lalsd(const char *uplo, f77_integer *smlsiz, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *d__, float *e, float *b, f77_integer CONST_REF ldb, float *rcond, f77_integer *rank, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lamc1(f77_integer *beta, f77_integer *t, f77_logical *rnd, f77_logical *ieee1);
//  static inline void lamc2(f77_integer *beta, f77_integer *t, f77_logical *rnd, float *eps, f77_integer *emin, float *rmin, f77_integer *emax, float *rmax);
//  static inline void lamc4(f77_integer *emin, float *start, f77_integer *base);
//  static inline void lamc5(f77_integer *beta, f77_integer CONST_REF p, f77_integer *emin, f77_logical *ieee, f77_integer *emax, float *rmax);
//  static inline void lamrg(f77_integer *n1, f77_integer *n2, float *a, f77_integer *dtrd1, f77_integer *dtrd2, f77_integer *index);
//  static inline void lanv2(float *a, float *b, float *c__, float *d__, float *rt1r, float *rt1i, float *rt2r, float *rt2i, float *cs, float *sn);
//  static inline void lapll(f77_integer CONST_REF n, float *x, f77_integer CONST_REF incx, float *y, f77_integer CONST_REF incy, float *ssmin);
//  static inline void lapmt(f77_logical *forwrd, f77_integer CONST_REF m, f77_integer CONST_REF n, float *x, f77_integer CONST_REF ldx, f77_integer CONST_REF k);
//  static inline void laqgb(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, float *ab, f77_integer CONST_REF ldab, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, const char *equed);
//  static inline void laqge(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *r__, float *c__, float *rowcnd, float *colcnd, float *amax, const char *equed);
//  static inline void laqp2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *offset, float *a, f77_integer CONST_REF lda, f77_integer *jpvt, float *tau, float *vn1, float *vn2, float *work);
//  static inline void laqps(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *offset, f77_integer *nb, f77_integer *kb, float *a, f77_integer CONST_REF lda, f77_integer *jpvt, float *tau, float *vn1, float *vn2, float *auxv, float *f, f77_integer CONST_REF ldf);
//  static inline void laqsb(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqsp(const char *uplo, f77_integer CONST_REF n, float *ap, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqsy(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *s, float *scond, float *amax, const char *equed);
//  static inline void laqtr(f77_logical *ltran, f77_logical *lf77_real, f77_integer CONST_REF n, float *t, f77_integer CONST_REF ldt, float *b, float *w, float *scale, float *x, float *work, f77_integer *info);
//  static inline void lar1v(f77_integer CONST_REF n, f77_integer *b1, f77_integer *bn, float *sigma, float *d__, float *l, float *ld, float *lld, float *gersch, float *z__, float *ztz, float *mingma, f77_integer *r__, f77_integer *isuppz, float *work);
//  static inline void lar2v(f77_integer CONST_REF n, float *x, float *y, float *z__, f77_integer CONST_REF incx, float *c__, float *s, f77_integer CONST_REF incc);
//  static inline void larf(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, float *v, f77_integer CONST_REF incv, float *tau, float *c__, f77_integer CONST_REF ldc, float *work);
//  static inline void larfb(const char *side, const char *trans, const char *direct, const char *storev, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *v, f77_integer CONST_REF ldv, float *t, f77_integer CONST_REF ldt, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF ldwork);
//  static inline void larfg(f77_integer CONST_REF n, float *alpha, float *x, f77_integer CONST_REF incx, float *tau);
//  static inline void larft(const char *direct, const char *storev, f77_integer CONST_REF n, f77_integer CONST_REF k, float *v, f77_integer CONST_REF ldv, float *tau, float *t, f77_integer CONST_REF ldt);
//  static inline void larfx(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, float *v, float *tau, float *c__, f77_integer CONST_REF ldc, float *work);
//  static inline void largv(f77_integer CONST_REF n, float *x, f77_integer CONST_REF incx, float *y, f77_integer CONST_REF incy, float *c__, f77_integer CONST_REF incc);
//  static inline void larnv(f77_integer *idist, f77_integer *iseed, f77_integer CONST_REF n, float *x);
//  static inline void larrb(f77_integer CONST_REF n, float *d__, float *l, float *ld, float *lld, f77_integer *ifirst, f77_integer *ilast, float *sigma, float *reltol, float *w, float *wgap, float *werr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void larre(f77_integer CONST_REF n, float *d__, float *e, float *tol, f77_integer *nsplit, f77_integer *isplit, f77_integer *m_out, float *w, float *woff, float *gersch, float *work, f77_integer *info);
//  static inline void larrf(f77_integer CONST_REF n, float *d__, float *l, float *ld, float *lld, f77_integer *ifirst, f77_integer *ilast, float *w, float *dplus, float *lplus, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void larrv(f77_integer CONST_REF n, float *d__, float *l, f77_integer *isplit, f77_integer CONST_REF m, float *w, f77_integer *iblock, float *gersch, float *tol, float *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lartg(float *f, float *g, float *cs, float *sn, float *r__);
//  static inline void lartv(f77_integer CONST_REF n, float *x, f77_integer CONST_REF incx, float *y, f77_integer CONST_REF incy, float *c__, float *s, f77_integer CONST_REF incc);
//  static inline void laruv(f77_integer *iseed, f77_integer CONST_REF n, float *x);
//  static inline void larz(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *l, float *v, f77_integer CONST_REF incv, float *tau, float *c__, f77_integer CONST_REF ldc, float *work);
//  static inline void larzb(const char *side, const char *trans, const char *direct, const char *storev, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, float *v, f77_integer CONST_REF ldv, float *t, f77_integer CONST_REF ldt, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF ldwork);
//  static inline void larzt(const char *direct, const char *storev, f77_integer CONST_REF n, f77_integer CONST_REF k, float *v, f77_integer CONST_REF ldv, float *tau, float *t, f77_integer CONST_REF ldt);
//  static inline void las2(float *f, float *g, float *h__, float *ssmin, float *ssmax);
//  static inline void lascl(const char *type__, f77_integer *kl, f77_integer *ku, float *cfrom, float *cto, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void lasd0(f77_integer CONST_REF n, f77_integer *sqre, float *d__, float *e, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF ldvt, f77_integer *smlsiz, f77_integer *iwork, float *work, f77_integer *info);
//  static inline void lasd1(f77_integer *nl, f77_integer *nr, f77_integer *sqre, float *d__, float *alpha, float *beta, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF ldvt, f77_integer *idxq, f77_integer *iwork, float *work, f77_integer *info);
//  static inline void lasd2(f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF k, float *d__, float *z__, float *alpha, float *beta, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF ldvt, float *dsigma, float *u2, f77_integer CONST_REF ldu2, float *vt2, f77_integer CONST_REF ldvt2, f77_integer *idxp, f77_integer *idx, f77_integer *idxc, f77_integer *idxq, f77_integer *coltyp, f77_integer *info);
//  static inline void lasd3(f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF k, float *d__, float *q, f77_integer CONST_REF ldq, float *dsigma, float *u, f77_integer CONST_REF ldu, float *u2, f77_integer CONST_REF ldu2, float *vt, f77_integer CONST_REF ldvt, float *vt2, f77_integer CONST_REF ldvt2, f77_integer *idxc, f77_integer *ctot, float *z__, f77_integer *info);
//  static inline void lasd4(f77_integer CONST_REF n, f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *sigma, float *work, f77_integer *info);
//  static inline void lasd5(f77_integer *i__, float *d__, float *z__, float *delta, float *rho, float *dsigma, float *work);
//  static inline void lasd6(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, float *d__, float *vf, float *vl, float *alpha, float *beta, f77_integer *idxq, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, float *givnum, f77_integer CONST_REF ldgnum, float *poles, float *difl, float *difr, float *z__, f77_integer CONST_REF k, float *c__, float *s, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lasd7(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF k, float *d__, float *z__, float *zw, float *vf, float *vfw, float *vl, float *vlw, float *alpha, float *beta, float *dsigma, f77_integer *idx, f77_integer *idxp, f77_integer *idxq, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, float *givnum, f77_integer CONST_REF ldgnum, float *c__, float *s, f77_integer *info);
//  static inline void lasd8(f77_integer *icompq, f77_integer CONST_REF k, float *d__, float *z__, float *vf, float *vl, float *difl, float *difr, f77_integer CONST_REF lddifr, float *dsigma, float *work, f77_integer *info);
//  static inline void lasd9(f77_integer *icompq, f77_integer CONST_REF ldu, f77_integer CONST_REF k, float *d__, float *z__, float *vf, float *vl, float *difl, float *difr, float *dsigma, float *work, f77_integer *info);
//  static inline void lasda(f77_integer *icompq, f77_integer *smlsiz, f77_integer CONST_REF n, f77_integer *sqre, float *d__, float *e, float *u, f77_integer CONST_REF ldu, float *vt, f77_integer CONST_REF k, float *difl, float *difr, float *z__, float *poles, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, f77_integer *perm, float *givnum, float *c__, float *s, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void lasdq(const char *uplo, f77_integer *sqre, f77_integer CONST_REF n, f77_integer CONST_REF ncvt, f77_integer CONST_REF nru, f77_integer CONST_REF ncc, float *d__, float *e, float *vt, f77_integer CONST_REF ldvt, float *u, f77_integer CONST_REF ldu, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void lasdt(f77_integer CONST_REF n, f77_integer *lvl, f77_integer *nd, f77_integer *inode, f77_integer *ndiml, f77_integer *ndimr, f77_integer *msub);
//  static inline void laset(const char *uplo, f77_integer CONST_REF m, f77_integer CONST_REF n, float *alpha, float *beta, float *a, f77_integer CONST_REF lda);
//  static inline void lasq1(f77_integer CONST_REF n, float *d__, float *e, float *work, f77_integer *info);
//  static inline void lasq2(f77_integer CONST_REF n, float *z__, f77_integer *info);
//  static inline void lasq3(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, float *dmin__, float *sigma, float *desig, float *qmax, f77_integer *nfail, f77_integer *iter, f77_integer *ndiv, f77_logical *ieee);
//  static inline void lasq4(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, f77_integer *n0in, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dn1, float *dn2, float *tau, f77_integer *ttype);
//  static inline void lasq5(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, float *tau, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dnm1, float *dnm2, f77_logical *ieee);
//  static inline void lasq6(f77_integer *i0, f77_integer *n0, float *z__, f77_integer *pp, float *dmin__, float *dmin1, float *dmin2, float *dn, float *dnm1, float *dnm2);
//  static inline void lasr(const char *side, const char *pivot, const char *direct, f77_integer CONST_REF m, f77_integer CONST_REF n, float *c__, float *s, float *a, f77_integer CONST_REF lda);
//  static inline void lasrt(const char *id, f77_integer CONST_REF n, float *d__, f77_integer *info);
//  static inline void lassq(f77_integer CONST_REF n, float *x, f77_integer CONST_REF incx, float *scale, float *sumsq);
//  static inline void lasv2(float *f, float *g, float *h__, float *ssmin, float *ssmax, float *snr, float *csr, float *snl, float *csl);
//  static inline void laswp(f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *k1, f77_integer *k2, f77_integer *ipiv, f77_integer CONST_REF incx);
//  static inline void lasy2(f77_logical *ltranl, f77_logical *ltranr, f77_integer *isgn, f77_integer *n1, f77_integer *n2, float *tl, f77_integer CONST_REF ldtl, float *tr, f77_integer CONST_REF ldtr, float *b, f77_integer CONST_REF ldb, float *scale, float *x, f77_integer CONST_REF ldx, float *xnorm, f77_integer *info);
//  static inline void lasyf(const char *uplo, f77_integer CONST_REF n, f77_integer *nb, f77_integer *kb, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *w, f77_integer CONST_REF ldw, f77_integer *info);
//  static inline void latbs(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *x, float *scale, float *cnorm, f77_integer *info);
//  static inline void latdf(f77_integer *ijob, f77_integer CONST_REF n, float *z__, f77_integer CONST_REF ldz, float *rhs, float *rdsum, float *rdscal, f77_integer *ipiv, f77_integer *jpiv);
//  static inline void latps(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer CONST_REF n, float *ap, float *x, float *scale, float *cnorm, f77_integer *info);
//  static inline void latrd(const char *uplo, f77_integer CONST_REF n, f77_integer *nb, float *a, f77_integer CONST_REF lda, float *e, float *tau, float *w, f77_integer CONST_REF ldw);
//  static inline void latrs(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *x, float *scale, float *cnorm, f77_integer *info);
//  static inline void latrz(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *l, float *a, f77_integer CONST_REF lda, float *tau, float *work);
//  static inline void latzm(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, float *v, f77_integer CONST_REF incv, float *tau, float *c1, float *c2, f77_integer CONST_REF ldc, float *work);
//  static inline void lauu2(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void lauum(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void opgtr(const char *uplo, f77_integer CONST_REF n, float *ap, float *tau, float *q, f77_integer CONST_REF ldq, float *work, f77_integer *info);
//  static inline void opmtr(const char *side, const char *uplo, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, float *ap, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void org2l(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void org2r(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void orgbr(const char *vect, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orghr(f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orgl2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void orglq(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orgql(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
  static inline void orgqr(f77_integer m, f77_integer n, f77_integer k, float *a, f77_integer lda, float *tau, float *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(sorgqr)(m, n, k, a, lda, tau, work, lwork, info);

  }
//  static inline void orgr2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer *info);
//  static inline void orgrq(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orgtr(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orm2l(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void orm2r(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void ormbr(const char *vect, const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormhr(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orml2(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void ormlq(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormql(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormqr(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormr2(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void ormr3(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer *info);
//  static inline void ormrq(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormrz(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormtr(const char *side, const char *uplo, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *c__, f77_integer CONST_REF ldc, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void pbcon(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbequ(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *s, float *scond, float *amax, f77_integer *info);
//  static inline void pbrfs(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *afb, f77_integer CONST_REF ldafb, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbstf(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, f77_integer *info);
//  static inline void pbsv(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void pbsvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *afb, f77_integer CONST_REF ldafb, const char *equed, float *s, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbtf2(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, f77_integer *info);
//  static inline void pbtrf(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, f77_integer *info);
//  static inline void pbtrs(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void pocon(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void poequ(f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *s, float *scond, float *amax, f77_integer *info);
//  static inline void porfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *af, f77_integer CONST_REF ldaf, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void posv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void posvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *af, f77_integer CONST_REF ldaf, const char *equed, float *s, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void potf2(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
  static inline void potrf(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info) {
   F77_FUNC(spotrf)(uplo, n, a, lda, info);
  }
//  static inline void potri(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void potrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ppcon(const char *uplo, f77_integer CONST_REF n, float *ap, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void ppequ(const char *uplo, f77_integer CONST_REF n, float *ap, float *s, float *scond, float *amax, f77_integer *info);
//  static inline void pprfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *afp, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void ppsv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ppsvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *afp, const char *equed, float *s, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void pptrf(const char *uplo, f77_integer CONST_REF n, float *ap, f77_integer *info);
//  static inline void pptri(const char *uplo, f77_integer CONST_REF n, float *ap, f77_integer *info);
//  static inline void pptrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ptcon(f77_integer CONST_REF n, float *d__, float *e, float *anorm, float *rcond, float *work, f77_integer *info);
//  static inline void pteqr(const char *compz, f77_integer CONST_REF n, float *d__, float *e, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void ptrfs(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *d__, float *e, float *df, float *ef, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *info);
//  static inline void ptsv(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *d__, float *e, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ptsvx(const char *fact, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *d__, float *e, float *df, float *ef, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *info);
//  static inline void pttrf(f77_integer CONST_REF n, float *d__, float *e, f77_integer *info);
//  static inline void pttrs(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *d__, float *e, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ptts2(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *d__, float *e, float *b, f77_integer CONST_REF ldb);
//  static inline void rscl(f77_integer CONST_REF n, float *sa, float *sx, f77_integer CONST_REF incx);
//  static inline void sbev(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void sbevd(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void sbevx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *q, f77_integer CONST_REF ldq, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sbgst(const char *vect, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer CONST_REF ldab, float *bb, f77_integer CONST_REF ldbb, float *x, f77_integer CONST_REF ldx, float *work, f77_integer *info);
//  static inline void sbgv(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer CONST_REF ldab, float *bb, f77_integer CONST_REF ldbb, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void sbgvd(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer CONST_REF ldab, float *bb, f77_integer CONST_REF ldbb, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void sbgvx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, float *ab, f77_integer CONST_REF ldab, float *bb, f77_integer CONST_REF ldbb, float *q, f77_integer CONST_REF ldq, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sbtrd(const char *vect, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *d__, float *e, float *q, f77_integer CONST_REF ldq, float *work, f77_integer *info);
//  static inline void spcon(const char *uplo, f77_integer CONST_REF n, float *ap, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void spev(const char *jobz, const char *uplo, f77_integer CONST_REF n, float *ap, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void spevd(const char *jobz, const char *uplo, f77_integer CONST_REF n, float *ap, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void spevx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, float *ap, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void spgst(f77_integer *itype, const char *uplo, f77_integer CONST_REF n, float *ap, float *bp, f77_integer *info);
//  static inline void spgv(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, float *ap, float *bp, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void spgvd(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, float *ap, float *bp, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void spgvx(f77_integer *itype, const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, float *ap, float *bp, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sprfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *afp, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void spsv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void spsvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *afp, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void sptrd(const char *uplo, f77_integer CONST_REF n, float *ap, float *d__, float *e, float *tau, f77_integer *info);
//  static inline void sptrf(const char *uplo, f77_integer CONST_REF n, float *ap, f77_integer *ipiv, f77_integer *info);
//  static inline void sptri(const char *uplo, f77_integer CONST_REF n, float *ap, f77_integer *ipiv, float *work, f77_integer *info);
//  static inline void sptrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void stebz(const char *range, const char *order, f77_integer CONST_REF n, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, float *d__, float *e, f77_integer *m_out, f77_integer *nsplit, float *w, f77_integer *iblock, f77_integer *isplit, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void stedc(const char *compz, f77_integer CONST_REF n, float *d__, float *e, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stegr(const char *jobz, const char *range, f77_integer CONST_REF n, float *d__, float *e, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stein(f77_integer CONST_REF n, float *d__, float *e, f77_integer CONST_REF m, float *w, f77_integer *iblock, f77_integer *isplit, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void steqr(const char *compz, f77_integer CONST_REF n, float *d__, float *e, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void sterf(f77_integer CONST_REF n, float *d__, float *e, f77_integer *info);
//  static inline void stev(const char *jobz, f77_integer CONST_REF n, float *d__, float *e, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *info);
//  static inline void stevd(const char *jobz, f77_integer CONST_REF n, float *d__, float *e, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stevr(const char *jobz, const char *range, f77_integer CONST_REF n, float *d__, float *e, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stevx(const char *jobz, const char *range, f77_integer CONST_REF n, float *d__, float *e, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sycon(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *anorm, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void syev(const char *jobz, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *w, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void syevd(const char *jobz, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *w, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void syevr(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void syevx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sygs2(f77_integer *itype, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void sygst(f77_integer *itype, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *info);
  static inline void sygv(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *w, float *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(ssygv)(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);
  }
//  static inline void sygvd(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *w, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void sygvx(f77_integer *itype, const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *vl, float *vu, f77_integer *il, f77_integer *iu, float *abstol, f77_integer *m_out, float *w, float *z__, f77_integer CONST_REF ldz, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void syrfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void sysv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void sysvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *rcond, float *ferr, float *berr, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void sytd2(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *d__, float *e, float *tau, f77_integer *info);
//  static inline void sytf2(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *info);
//  static inline void sytrd(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *d__, float *e, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void sytrf(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void sytri(const char *uplo, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *work, f77_integer *info);
//  static inline void sytrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, f77_integer *ipiv, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void tbcon(const char *norm, const char *uplo, const char *diag, f77_integer CONST_REF n, f77_integer *kd, float *ab, f77_integer CONST_REF ldab, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tbrfs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tbtrs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, float *ab, f77_integer CONST_REF ldab, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void tgevc(const char *side, const char *howmny, f77_logical *select, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, f77_integer *mm, f77_integer *m_out, float *work, f77_integer *info);
//  static inline void tgex2(f77_logical CONST_REF wantq, f77_logical CONST_REF wantz, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *q, f77_integer CONST_REF ldq, float *z__, f77_integer CONST_REF ldz, f77_integer *j1, f77_integer *n1, f77_integer *n2, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void tgexc(f77_logical CONST_REF wantq, f77_logical CONST_REF wantz, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *q, f77_integer CONST_REF ldq, float *z__, f77_integer CONST_REF ldz, f77_integer *ifst, f77_integer *ilst, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void tgsen(f77_integer *ijob, f77_logical CONST_REF wantq, f77_logical CONST_REF wantz, f77_logical *select, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *alphar, float *alphai, float *beta, float *q, f77_integer CONST_REF ldq, float *z__, f77_integer CONST_REF ldz, f77_integer *m_out, float *pl, float *pr, float *dif, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void tgsja(const char *jobu, const char *jobv, const char *jobq, f77_integer CONST_REF m, f77_integer CONST_REF p, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *tola, float *tolb, float *alpha, float *beta, float *u, f77_integer CONST_REF ldu, float *v, f77_integer CONST_REF ldv, float *q, f77_integer CONST_REF ldq, float *work, f77_integer *ncycle, f77_integer *info);
//  static inline void tgsna(const char *job, const char *howmny, f77_logical *select, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, float *s, float *dif, f77_integer *mm, f77_integer *m_out, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void tgsy2(const char *trans, f77_integer *ijob, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *c__, f77_integer CONST_REF ldc, float *d__, f77_integer CONST_REF ldd, float *e, f77_integer CONST_REF lde, float *f, f77_integer CONST_REF ldf, float *scale, float *rdsum, float *rdscal, f77_integer *iwork, f77_integer *pq, f77_integer *info);
//  static inline void tgsyl(const char *trans, f77_integer *ijob, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *c__, f77_integer CONST_REF ldc, float *d__, f77_integer CONST_REF ldd, float *e, f77_integer CONST_REF lde, float *f, f77_integer CONST_REF ldf, float *scale, float *dif, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void tpcon(const char *norm, const char *uplo, const char *diag, f77_integer CONST_REF n, float *ap, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tprfs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void tptri(const char *uplo, const char *diag, f77_integer CONST_REF n, float *ap, f77_integer *info);
//  static inline void tptrs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *ap, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void trcon(const char *norm, const char *uplo, const char *diag, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *rcond, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void trevc(const char *side, const char *howmny, f77_logical *select, f77_integer CONST_REF n, float *t, f77_integer CONST_REF ldt, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, f77_integer *mm, f77_integer *m_out, float *work, f77_integer *info);
//  static inline void trexc(const char *compq, f77_integer CONST_REF n, float *t, f77_integer CONST_REF ldt, float *q, f77_integer CONST_REF ldq, f77_integer *ifst, f77_integer *ilst, float *work, f77_integer *info);
//  static inline void trrfs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *x, f77_integer CONST_REF ldx, float *ferr, float *berr, float *work, f77_integer *iwork, f77_integer *info);
//  static inline void trsen(const char *job, const char *compq, f77_logical *select, f77_integer CONST_REF n, float *t, f77_integer CONST_REF ldt, float *q, f77_integer CONST_REF ldq, float *wr, float *wi, f77_integer *m_out, float *s, float *sep, float *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void trsna(const char *job, const char *howmny, f77_logical *select, f77_integer CONST_REF n, float *t, f77_integer CONST_REF ldt, float *vl, f77_integer CONST_REF ldvl, float *vr, f77_integer CONST_REF ldvr, float *s, float *sep, f77_integer *mm, f77_integer *m_out, float *work, f77_integer CONST_REF ldwork, f77_integer *iwork, f77_integer *info);
//  static inline void trsyl(const char *trana, const char *tranb, f77_integer *isgn, f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, float *c__, f77_integer CONST_REF ldc, float *scale, f77_integer *info);
//  static inline void trti2(const char *uplo, const char *diag, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void trtri(const char *uplo, const char *diag, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void trtrs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, float *a, f77_integer CONST_REF lda, float *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void tzrqf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, f77_integer *info);
//  static inline void tzrzf(f77_integer CONST_REF m, f77_integer CONST_REF n, float *a, f77_integer CONST_REF lda, float *tau, float *work, f77_integer CONST_REF lwork, f77_integer *info);
//  f77_integer icmax1(f77_integer CONST_REF n, f77_complex *cx, f77_integer CONST_REF incx);
//  f77_integer ieeeck(f77_integer CONST_REF ispec, f77_real *zero, f77_real *one);
//  f77_integer ilaenv(f77_integer CONST_REF ispec, const char *name__, const char *opts, f77_integer *n1, f77_integer *n2, f77_integer *n3, f77_integer *n4, f77_str_len name_len, f77_str_len opts_len);
//  f77_integer izmax1(f77_integer CONST_REF n, floatcomplex *cx, f77_integer CONST_REF incx);

}; //CppLapack
template<>
class CppLapack<double> {
 public:
  CppLapack() {
    double fake_matrix[64];
    double fake_workspace;
    double fake_vector;
    f77_integer fake_pivots;
    f77_integer fake_info;
      
    /* TODO: This may want to be ilaenv */
    this->getri(1, (double *)fake_matrix, 1, &fake_pivots, &fake_workspace,
        -1, &fake_info);
    this->getri_block_size = int(fake_workspace);
      
    this->geqrf(1, 1, (double *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
        &fake_info);
    this->geqrf_block_size = int(fake_workspace);
      
    this->orgqr(1, 1, 1, (double *)fake_matrix, 1, &fake_vector, &fake_workspace, -1,
         &fake_info);
    this->orgqr_block_size = int(fake_workspace);
      
    this->geqrf_dorgqr_block_size =
         std::max(this->geqrf_block_size, this->orgqr_block_size);
 
  }
  static int getri_block_size;
  static int geqrf_block_size;
  static int orgqr_block_size;
  static int geqrf_dorgqr_block_size;


  static inline void bdsdc(const char *uplo, const char *compq, f77_integer CONST_REF n, double *d__, double *e, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF ldvt, double *q, f77_integer *iq, double *work, f77_integer *iwork, f77_integer *info) {
     F77_FUNC(dbdsdc)(uplo, compq, n, d__, e, u, ldu, vt, ldvt, q, iq, work, iwork, info); 
  }
  static inline void bdsqr(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF ncvt, f77_integer CONST_REF nru, f77_integer CONST_REF ncc, double *d__, double *e, double *vt, f77_integer CONST_REF ldvt, double *u, f77_integer CONST_REF ldu, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info) {
     F77_FUNC(dbdsqr)(uplo, n, ncvt, nru, ncc, d__, e, vt, ldvt, u, ldu, c__, ldc, work, info);
  }
  static inline void disna(const char *job, f77_integer CONST_REF m, f77_integer CONST_REF n, double *d__, double *sep, f77_integer *info) {
     F77_FUNC(ddisna)(job, m, n, d__, sep, info);
  }
  static inline void gbbrd(const char *vect, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF ncc, f77_integer *kl, f77_integer *ku, double *ab, f77_integer CONST_REF ldab, double *d__, double *e, double *q, f77_integer CONST_REF ldq, double *pt, f77_integer CONST_REF ldpt, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info) {
     F77_FUNC(dgbbrd)(vect, m, n, ncc, kl, ku, ab, ldab, d__, e, q, ldq, pt, ldpt, c__, ldc, work, info);
  }
  static inline void gbcon(const char *norm, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, double *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info) {
     F77_FUNC(dgbcon)(norm, n, kl, ku, ab, ldab, ipiv, anorm, rcond, work, iwork, info);
  }
  static inline void gbequ(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, double *ab, f77_integer CONST_REF ldab, double *r__, double *c__, double *rowcnd, double *colcnd, double *amax, f77_integer *info) {
     F77_FUNC(dgbequ)(m, n, kl, ku, ab, ldab, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gbrfs(const char *trans, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *afb, f77_integer CONST_REF ldafb, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer*iwork, f77_integer *info) {
    F77_FUNC(dgbrfs)(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  }
  static inline void gbsv(f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info) {
    F77_FUNC(dgbsv)(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);
  }
  static inline void gbsvx(const char *fact, const char *trans, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *afb, f77_integer CONST_REF ldafb, f77_integer *ipiv, const char *equed, double *r__, double *c__, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(dgbsvx)(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r__, c__, b, ldb, x, ldx, rcond, ferr, berr, work, iwork, info);
  }
  static inline void gbtf2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, double *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(dgbtf2)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrf(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, double *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(dgbtrf)(m, n, kl, ku, ab, ldab, ipiv, info);
  }
  static inline void gbtrs(const char *trans, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info) {
    F77_FUNC(dgbtrs)(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info);

  }
  static inline void gebak(const char *job, const char *side, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *scale, f77_integer CONST_REF m, double *v, f77_integer CONST_REF ldv, f77_integer *info) {
    F77_FUNC(dgebak)(job, side, n, ilo, ihi, scale, m, v, ldv, info);
  }
  static inline void gebal(const char *job, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ilo, f77_integer *ihi, double *scale, f77_integer *info) {
    F77_FUNC(dgebal)(job, n, a, lda, ilo, ihi, scale, info);

  }
  static inline void gebd2(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *d__, double *e, double *tauq, double *taup, double *work, f77_integer *info) {
    F77_FUNC(dgebd2)(m, n, a, lda, d__, e, tauq, taup, work, info);
  }
  static inline void gebrd(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *d__, double *e, double *tauq, double *taup, double *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(dgebrd)(m, n, a, lda, d__, e, tauq, taup, work, lwork, info);
  }
  static inline void gecon(const char *norm, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(dgecon)(norm, n, a, lda, anorm, rcond, work, iwork, info);
  }
  static inline void geequ(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *r__, double *c__, double *rowcnd, double *colcnd, double *amax, f77_integer *info) {
    F77_FUNC(dgeequ)(m, n, a, lda, r__, c__, rowcnd, colcnd, amax, info);
  }
  static inline void gees(const char *jobvs, const char *sort, f77_logical_func select, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *sdim, double *wr, double *wi, double *vs, f77_integer CONST_REF ldvs, double *work, f77_integer CONST_REF lwork, unsigned int *bwork, f77_integer *info) {
    F77_FUNC(dgees)(jobvs, sort, select, n, a, lda, sdim, wr, wi, vs, ldvs, work, lwork, bwork, info);
  }
  static inline void geesx(const char *jobvs, const char *sort, f77_logical_func select, const char *sense, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *sdim, double *wr, double *wi, double *vs, f77_integer CONST_REF ldvs, double *rconde, double *rcondv, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, unsigned int *bwork, f77_integer *info) {
    F77_FUNC(dgeesx)(jobvs, sort, select, sense, n, a, lda, sdim, wr, wi, vs, ldvs, rconde, rcondv, work, lwork, iwork, liwork, bwork, info);
  }
  static inline void geev(const char *jobvl, const char *jobvr, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *wr, double *wi, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, double *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(dgeev)(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
  }
  static inline void geevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *wr, double *wi, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, f77_integer *ilo, f77_integer *ihi, double *scale, double *abnrm, double *rconde, double *rcondv, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(dgeevx)(balanc, jobvl, jobvr, sense, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, ilo, ihi, scale, abnrm, rconde, rcondv, work,lwork, iwork, info);
  }
  static inline void gegs(const char *jobvsl, const char *jobvsr, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *vsl, f77_integer CONST_REF ldvsl, double *vsr, f77_integer CONST_REF ldvsr, double *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(dgegs)(jobvsl, jobvsr, n, a, lda, b, ldb, alphar, alphai, beta, vsl, ldvsl, vsr, ldvsr, work, lwork, info);
  }
  static inline void gegv(const char *jobvl, const char *jobvr, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, double *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(dgegv)(jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);
  }
//  static inline void gehd2(f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void gehrd(f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gelq2(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void gelqf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gels(const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gelsd(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *s, double *rcond, f77_integer *rank, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void gelss(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *s, double *rcond, f77_integer *rank, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gelsx(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *jpvt, double *rcond, f77_integer *rank, double *work, f77_integer *info);
//  static inline void gelsy(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *jpvt, double *rcond, f77_integer *rank, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void geql2(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void geqlf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
// static inline void geqp3(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *jpvt, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void geqpf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *jpvt, double *tau, double *work, f77_integer *info);
//  static inline void geqr2(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
  static inline void geqrf(f77_integer m, f77_integer n, double *a, f77_integer lda, double *tau, double *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(dgeqrf)(m, n, a, lda, tau, work, lwork, info);
  }
  static inline void gerfs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(dgerfs)(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr, work, iwork, info);
  
  }
//  static inline void gerq2(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void gerqf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gesc2(f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *rhs, f77_integer *ipiv, f77_integer *jpiv, double *scale);
  static inline void gesdd(const char *jobz, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *s, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF ldvt, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info) {
    F77_FUNC(dgesdd)(jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info);
  }
  static inline void gesv(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info) {
    F77_FUNC(dgesv)(n, nrhs, a, lda, ipiv, b, ldb, info);
  }
//  static inline void gesvd(const char *jobu, const char *jobvt, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *s, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF ldvt, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gesvx(const char *fact, const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, const char *equed, double *r__, double *c__, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void getc2(f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *jpiv, f77_integer *info);
//  static inline void getf2(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *info);
  static inline void getrf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *info) {
    F77_FUNC(dgetrf)(m, n, a, lda, ipiv, info); 
  }
  static inline void getri(f77_integer n, double *a, f77_integer lda, f77_integer *ipiv, double *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(dgetri)(n, a, lda, ipiv, work, lwork, info);
  }
//  static inline void getrs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ggbak(const char *job, const char *side, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *lscale, double *rscale, f77_integer CONST_REF m, double *v, f77_integer CONST_REF ldv, f77_integer *info);
//  static inline void ggbal(const char *job, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *ilo, f77_integer *ihi, double *lscale, double *rscale, double *work, f77_integer *info);
//  static inline void gges(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *sdim, double *alphar, double *alphai, double *beta, double *vsl, f77_integer CONST_REF ldvsl, double *vsr, f77_integer CONST_REF ldvsr, double *work, f77_integer CONST_REF lwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggesx(const char *jobvsl, const char *jobvsr, const char *sort, f77_logical_func delctg, const char *sense, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *sdim, double *alphar, double *alphai, double *beta, double *vsl, f77_integer CONST_REF ldvsl, double *vsr, f77_integer CONST_REF ldvsr, double *rconde, double *rcondv, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggev(const char *jobvl, const char *jobvr, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggevx(const char *balanc, const char *jobvl, const char *jobvr, const char *sense, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, f77_integer *ilo, f77_integer *ihi, double *lscale, double *rscale, double *abnrm, double *bbnrm, double *rconde, double *rcondv, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_logical *bwork, f77_integer *info);
//  static inline void ggglm(f77_integer CONST_REF n, f77_integer CONST_REF m, f77_integer CONST_REF p, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *d__, double *x, double *y, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void gghrd(const char *compq, const char *compz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *q, f77_integer CONST_REF ldq, double *z__, f77_integer CONST_REF ldz, f77_integer *info);
//  static inline void gglse(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF p, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *c__, double *d__, double *x, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggqrf(f77_integer CONST_REF n, f77_integer CONST_REF m, f77_integer CONST_REF p, double *a, f77_integer CONST_REF lda, double *taua, double *b, f77_integer CONST_REF ldb, double *taub, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggrqf(f77_integer CONST_REF m, f77_integer CONST_REF p, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *taua, double *b, f77_integer CONST_REF ldb, double *taub, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ggsvd(const char *jobu, const char *jobv, const char *jobq, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF p, f77_integer CONST_REF k, f77_integer *l, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alpha, double *beta, double *u, f77_integer CONST_REF ldu, double *v, f77_integer CONST_REF ldv, double *q, f77_integer CONST_REF ldq, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void ggsvp(const char *jobu, const char *jobv, const char *jobq, f77_integer CONST_REF m, f77_integer CONST_REF p, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *tola, double *tolb, f77_integer CONST_REF k, f77_integer *l, double *u, f77_integer CONST_REF ldu, double *v, f77_integer CONST_REF ldv, double *q, f77_integer CONST_REF ldq, f77_integer *iwork, double *tau, double *work, f77_integer *info);
//  static inline void gtcon(const char *norm, f77_integer CONST_REF n, double *dl, double *d__, double *du, double *du2, f77_integer *ipiv, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void gtrfs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *dl, double *d__, double *du, double *dlf, double *df, double *duf, double *du2, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void gtsv(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *dl, double *d__, double *du, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void gtsvx(const char *fact, const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *dl, double *d__, double *du, double *dlf, double *df, double *duf, double *du2, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void gttrf(f77_integer CONST_REF n, double *dl, double *d__, double *du, double *du2, f77_integer *ipiv, f77_integer *info);
//  static inline void gttrs(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *dl, double *d__, double *du, double *du2, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void gtts2(f77_integer *itrans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *dl, double *d__, double *du, double *du2, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb);
//  static inline void hgeqz(const char *job, const char *compq, const char *compz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *q, f77_integer CONST_REF ldq, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void hsein(const char *side, const char *eigsrc, const char *initv, f77_logical *select, f77_integer CONST_REF n, double *h__, f77_integer CONST_REF ldh, double *wr, double *wi, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, f77_integer *mm, f77_integer *m_out, double *work, f77_integer *ifaill, f77_integer *ifailr, f77_integer *info);
//  static inline void hseqr(const char *job, const char *compz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *h__, f77_integer CONST_REF ldh, double *wr, double *wi, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void labad(double *small, double *large);
//  static inline void labrd(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *nb, double *a, f77_integer CONST_REF lda, double *d__, double *e, double *tauq, double *taup, double *x, f77_integer CONST_REF ldx, double *y, f77_integer CONST_REF ldy);
//  static inline void lacon(f77_integer CONST_REF n, double *v, double *x, f77_integer *isgn, double *est, f77_integer *kase);
//  static inline void lacpy(const char *uplo, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb);
//  static inline void ladiv(double *a, double *b, double *c__, double *d__, double *p, double *q);
//  static inline void lae2(double *a, double *b, double *c__, double *rt1, double *rt2);
//  static inline void laebz(f77_integer *ijob, f77_integer *nitmax, f77_integer CONST_REF n, f77_integer *mmax, f77_integer *minp, f77_integer *nbmin, double *abstol, double *reltol, double *pivmin, double *d__, double *e, double *e2, f77_integer *nval, double *ab, double *c__, f77_integer *mout, f77_integer *nab, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed0(f77_integer *icompq, f77_integer *qsiz, f77_integer CONST_REF n, double *d__, double *e, double *q, f77_integer CONST_REF ldq, double *qstore, f77_integer CONST_REF ldqs, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed1(f77_integer CONST_REF n, double *d__, double *q, f77_integer CONST_REF ldq, f77_integer *indxq, double *rho, f77_integer *cutpnt, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed2(f77_integer CONST_REF k, f77_integer CONST_REF n, f77_integer *n1, double *d__, double *q, f77_integer CONST_REF ldq, f77_integer *indxq, double *rho, double *z__, double *dlamda, double *w, double *q2, f77_integer *indx, f77_integer *indxc, f77_integer *indxp, f77_integer *coltyp, f77_integer *info);
//  static inline void laed3(f77_integer CONST_REF k, f77_integer CONST_REF n, f77_integer *n1, double *d__, double *q, f77_integer CONST_REF ldq, double *rho, double *dlamda, double *q2, f77_integer *indx, f77_integer *ctot, double *w, double *s, f77_integer *info);
//  static inline void laed4(f77_integer CONST_REF n, f77_integer *i__, double *d__, double *z__, double *delta, double *rho, double *dlam, f77_integer *info);
//  static inline void laed5(f77_integer *i__, double *d__, double *z__, double *delta, double *rho, double *dlam);
//  static inline void laed6(f77_integer *kniter, f77_logical *orgati, double *rho, double *d__, double *z__, double *finit, double *tau, f77_integer *info);
//  static inline void laed7(f77_integer *icompq, f77_integer CONST_REF n, f77_integer *qsiz, f77_integer *tlvls, f77_integer *curlvl, f77_integer *curpbm, double *d__, double *q, f77_integer CONST_REF ldq, f77_integer *indxq, double *rho, f77_integer *cutpnt, double *qstore, f77_integer *qptr, f77_integer *prmptr, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, double *givnum, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void laed8(f77_integer *icompq, f77_integer CONST_REF k, f77_integer CONST_REF n, f77_integer *qsiz, double *d__, double *q, f77_integer CONST_REF ldq, f77_integer *indxq, double *rho, f77_integer *cutpnt, double *z__, double *dlamda, double *q2, f77_integer CONST_REF ldq2, double *w, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, double *givnum, f77_integer *indxp, f77_integer *indx, f77_integer *info);
//  static inline void laed9(f77_integer CONST_REF k, f77_integer *kstart, f77_integer *kstop, f77_integer CONST_REF n, double *d__, double *q, f77_integer CONST_REF ldq, double *rho, double *dlamda, double *w, double *s, f77_integer CONST_REF lds, f77_integer *info);
//  static inline void laeda(f77_integer CONST_REF n, f77_integer *tlvls, f77_integer *curlvl, f77_integer *curpbm, f77_integer *prmptr, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, double *givnum, double *q, f77_integer *qptr, double *z__, double *ztemp, f77_integer *info);
//  static inline void laein(f77_logical *rightv, f77_logical *noinit, f77_integer CONST_REF n, double *h__, f77_integer CONST_REF ldh, double *wr, double *wi, double *vr, double *vi, double *b, f77_integer CONST_REF ldb, double *work, double *eps3, double *smlnum, double *bignum, f77_integer *info);
//  static inline void laev2(double *a, double *b, double *c__, double *rt1, double *rt2, double *cs1, double *sn1);
//  static inline void laexc(f77_logical CONST_REF wantq, f77_integer CONST_REF n, double *t, f77_integer CONST_REF ldt, double *q, f77_integer CONST_REF ldq, f77_integer *j1, f77_integer *n1, f77_integer *n2, double *work, f77_integer *info);
//  static inline void lag2(double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *safmin, double *scale1, double *scale2, double *wr1, double *wr2, double *wi);
//  static inline void lags2(f77_logical *upper, double *a1, double *a2, double *a3, double *b1, double *b2, double *b3, double *csu, double *snu, double *csv, double *snv, double *csq, double *snq);
//  static inline void lagtf(f77_integer CONST_REF n, double *a, double *lambda, double *b, double *c__, double *tol, double *d__, f77_integer *in, f77_integer *info);
//  static inline void lagtm(const char *trans, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *alpha, double *dl, double *d__, double *du, double *x, f77_integer CONST_REF ldx, double *beta, double *b, f77_integer CONST_REF ldb);
//  static inline void lagts(f77_integer *job, f77_integer CONST_REF n, double *a, double *b, double *c__, double *d__, f77_integer *in, double *y, double *tol, f77_integer *info);
//  static inline void lagv2(double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *csl, double *snl, double *csr, double *snr);
//  static inline void lahqr(f77_logical CONST_REF wantt, f77_logical CONST_REF wantz, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *h__, f77_integer CONST_REF ldh, double *wr, double *wi, f77_integer *iloz, f77_integer *ihiz, double *z__, f77_integer CONST_REF ldz, f77_integer *info);
//  static inline void lahrd(f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *nb, double *a, f77_integer CONST_REF lda, double *tau, double *t, f77_integer CONST_REF ldt, double *y, f77_integer CONST_REF ldy);
//  static inline void laic1(f77_integer *job, f77_integer *j, double *x, double *sest, double *w, double *gamma, double *sestpr, double *s, double *c__);
//  static inline void laln2(f77_logical *ltrans, f77_integer *na, f77_integer *nw, double *smin, double *ca, double *a, f77_integer CONST_REF lda, double *d1, double *d2, double *b, f77_integer CONST_REF ldb, double *wr, double *wi, double *x, f77_integer CONST_REF ldx, double *scale, double *xnorm, f77_integer *info);
//  static inline void lals0(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF nrhs, double *b, f77_integer CONST_REF ldb, double *bx, f77_integer CONST_REF ldbx, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, double *givnum, f77_integer CONST_REF ldgnum, double *poles, double *difl, double *difr, double *z__, f77_integer CONST_REF k, double *c__, double *s, double *work, f77_integer *info);
//  static inline void lalsa(f77_integer *icompq, f77_integer *smlsiz, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *b, f77_integer CONST_REF ldb, double *bx, f77_integer CONST_REF ldbx, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF k, double *difl, double *difr, double *z__, double *poles, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, f77_integer *perm, double *givnum, double *c__, double *s, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void lalsd(const char *uplo, f77_integer *smlsiz, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *d__, double *e, double *b, f77_integer CONST_REF ldb, double *rcond, f77_integer *rank, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void lamc1(f77_integer *beta, f77_integer *t, f77_logical *rnd, f77_logical *ieee1);
//  static inline void lamc2(f77_integer *beta, f77_integer *t, f77_logical *rnd, double *eps, f77_integer *emin, double *rmin, f77_integer *emax, double *rmax);
//  static inline void lamc4(f77_integer *emin, double *start, f77_integer *base);
//  static inline void lamc5(f77_integer *beta, f77_integer CONST_REF p, f77_integer *emin, f77_logical *ieee, f77_integer *emax, double *rmax);
//  static inline void lamrg(f77_integer *n1, f77_integer *n2, double *a, f77_integer *dtrd1, f77_integer *dtrd2, f77_integer *index);
//  static inline void lanv2(double *a, double *b, double *c__, double *d__, double *rt1r, double *rt1i, double *rt2r, double *rt2i, double *cs, double *sn);
//  static inline void lapll(f77_integer CONST_REF n, double *x, f77_integer CONST_REF incx, double *y, f77_integer CONST_REF incy, double *ssmin);
//  static inline void lapmt(f77_logical *forwrd, f77_integer CONST_REF m, f77_integer CONST_REF n, double *x, f77_integer CONST_REF ldx, f77_integer CONST_REF k);
//  static inline void laqgb(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *kl, f77_integer *ku, double *ab, f77_integer CONST_REF ldab, double *r__, double *c__, double *rowcnd, double *colcnd, double *amax, const char *equed);
//  static inline void laqge(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *r__, double *c__, double *rowcnd, double *colcnd, double *amax, const char *equed);
//  static inline void laqp2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *offset, double *a, f77_integer CONST_REF lda, f77_integer *jpvt, double *tau, double *vn1, double *vn2, double *work);
//  static inline void laqps(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *offset, f77_integer *nb, f77_integer *kb, double *a, f77_integer CONST_REF lda, f77_integer *jpvt, double *tau, double *vn1, double *vn2, double *auxv, double *f, f77_integer CONST_REF ldf);
//  static inline void laqsb(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *s, double *scond, double *amax, const char *equed);
//  static inline void laqsp(const char *uplo, f77_integer CONST_REF n, double *ap, double *s, double *scond, double *amax, const char *equed);
//  static inline void laqsy(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *s, double *scond, double *amax, const char *equed);
//  static inline void laqtr(f77_logical *ltran, f77_logical *lf77_real, f77_integer CONST_REF n, double *t, f77_integer CONST_REF ldt, double *b, double *w, double *scale, double *x, double *work, f77_integer *info);
//  static inline void lar1v(f77_integer CONST_REF n, f77_integer *b1, f77_integer *bn, double *sigma, double *d__, double *l, double *ld, double *lld, double *gersch, double *z__, double *ztz, double *mingma, f77_integer *r__, f77_integer *isuppz, double *work);
//  static inline void lar2v(f77_integer CONST_REF n, double *x, double *y, double *z__, f77_integer CONST_REF incx, double *c__, double *s, f77_integer CONST_REF incc);
//  static inline void larf(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, double *v, f77_integer CONST_REF incv, double *tau, double *c__, f77_integer CONST_REF ldc, double *work);
//  static inline void larfb(const char *side, const char *trans, const char *direct, const char *storev, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *v, f77_integer CONST_REF ldv, double *t, f77_integer CONST_REF ldt, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF ldwork);
//  static inline void larfg(f77_integer CONST_REF n, double *alpha, double *x, f77_integer CONST_REF incx, double *tau);
//  static inline void larft(const char *direct, const char *storev, f77_integer CONST_REF n, f77_integer CONST_REF k, double *v, f77_integer CONST_REF ldv, double *tau, double *t, f77_integer CONST_REF ldt);
//  static inline void larfx(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, double *v, double *tau, double *c__, f77_integer CONST_REF ldc, double *work);
//  static inline void largv(f77_integer CONST_REF n, double *x, f77_integer CONST_REF incx, double *y, f77_integer CONST_REF incy, double *c__, f77_integer CONST_REF incc);
//  static inline void larnv(f77_integer *idist, f77_integer *iseed, f77_integer CONST_REF n, double *x);
//  static inline void larrb(f77_integer CONST_REF n, double *d__, double *l, double *ld, double *lld, f77_integer *ifirst, f77_integer *ilast, double *sigma, double *reltol, double *w, double *wgap, double *werr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void larre(f77_integer CONST_REF n, double *d__, double *e, double *tol, f77_integer *nsplit, f77_integer *isplit, f77_integer *m_out, double *w, double *woff, double *gersch, double *work, f77_integer *info);
//  static inline void larrf(f77_integer CONST_REF n, double *d__, double *l, double *ld, double *lld, f77_integer *ifirst, f77_integer *ilast, double *w, double *dplus, double *lplus, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void larrv(f77_integer CONST_REF n, double *d__, double *l, f77_integer *isplit, f77_integer CONST_REF m, double *w, f77_integer *iblock, double *gersch, double *tol, double *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void lartg(double *f, double *g, double *cs, double *sn, double *r__);
//  static inline void lartv(f77_integer CONST_REF n, double *x, f77_integer CONST_REF incx, double *y, f77_integer CONST_REF incy, double *c__, double *s, f77_integer CONST_REF incc);
//  static inline void laruv(f77_integer *iseed, f77_integer CONST_REF n, double *x);
//  static inline void larz(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *l, double *v, f77_integer CONST_REF incv, double *tau, double *c__, f77_integer CONST_REF ldc, double *work);
//  static inline void larzb(const char *side, const char *trans, const char *direct, const char *storev, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, double *v, f77_integer CONST_REF ldv, double *t, f77_integer CONST_REF ldt, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF ldwork);
//  static inline void larzt(const char *direct, const char *storev, f77_integer CONST_REF n, f77_integer CONST_REF k, double *v, f77_integer CONST_REF ldv, double *tau, double *t, f77_integer CONST_REF ldt);
//  static inline void las2(double *f, double *g, double *h__, double *ssmin, double *ssmax);
//  static inline void lascl(const char *type__, f77_integer *kl, f77_integer *ku, double *cfrom, double *cto, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void lasd0(f77_integer CONST_REF n, f77_integer *sqre, double *d__, double *e, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF ldvt, f77_integer *smlsiz, f77_integer *iwork, double *work, f77_integer *info);
//  static inline void lasd1(f77_integer *nl, f77_integer *nr, f77_integer *sqre, double *d__, double *alpha, double *beta, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF ldvt, f77_integer *idxq, f77_integer *iwork, double *work, f77_integer *info);
//  static inline void lasd2(f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF k, double *d__, double *z__, double *alpha, double *beta, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF ldvt, double *dsigma, double *u2, f77_integer CONST_REF ldu2, double *vt2, f77_integer CONST_REF ldvt2, f77_integer *idxp, f77_integer *idx, f77_integer *idxc, f77_integer *idxq, f77_integer *coltyp, f77_integer *info);
//  static inline void lasd3(f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF k, double *d__, double *q, f77_integer CONST_REF ldq, double *dsigma, double *u, f77_integer CONST_REF ldu, double *u2, f77_integer CONST_REF ldu2, double *vt, f77_integer CONST_REF ldvt, double *vt2, f77_integer CONST_REF ldvt2, f77_integer *idxc, f77_integer *ctot, double *z__, f77_integer *info);
//  static inline void lasd4(f77_integer CONST_REF n, f77_integer *i__, double *d__, double *z__, double *delta, double *rho, double *sigma, double *work, f77_integer *info);
//  static inline void lasd5(f77_integer *i__, double *d__, double *z__, double *delta, double *rho, double *dsigma, double *work);
//  static inline void lasd6(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, double *d__, double *vf, double *vl, double *alpha, double *beta, f77_integer *idxq, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, double *givnum, f77_integer CONST_REF ldgnum, double *poles, double *difl, double *difr, double *z__, f77_integer CONST_REF k, double *c__, double *s, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void lasd7(f77_integer *icompq, f77_integer *nl, f77_integer *nr, f77_integer *sqre, f77_integer CONST_REF k, double *d__, double *z__, double *zw, double *vf, double *vfw, double *vl, double *vlw, double *alpha, double *beta, double *dsigma, f77_integer *idx, f77_integer *idxp, f77_integer *idxq, f77_integer *perm, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, double *givnum, f77_integer CONST_REF ldgnum, double *c__, double *s, f77_integer *info);
//  static inline void lasd8(f77_integer *icompq, f77_integer CONST_REF k, double *d__, double *z__, double *vf, double *vl, double *difl, double *difr, f77_integer CONST_REF lddifr, double *dsigma, double *work, f77_integer *info);
//  static inline void lasd9(f77_integer *icompq, f77_integer CONST_REF ldu, f77_integer CONST_REF k, double *d__, double *z__, double *vf, double *vl, double *difl, double *difr, double *dsigma, double *work, f77_integer *info);
//  static inline void lasda(f77_integer *icompq, f77_integer *smlsiz, f77_integer CONST_REF n, f77_integer *sqre, double *d__, double *e, double *u, f77_integer CONST_REF ldu, double *vt, f77_integer CONST_REF k, double *difl, double *difr, double *z__, double *poles, f77_integer *givptr, f77_integer *givcol, f77_integer CONST_REF ldgcol, f77_integer *perm, double *givnum, double *c__, double *s, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void lasdq(const char *uplo, f77_integer *sqre, f77_integer CONST_REF n, f77_integer CONST_REF ncvt, f77_integer CONST_REF nru, f77_integer CONST_REF ncc, double *d__, double *e, double *vt, f77_integer CONST_REF ldvt, double *u, f77_integer CONST_REF ldu, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void lasdt(f77_integer CONST_REF n, f77_integer *lvl, f77_integer *nd, f77_integer *inode, f77_integer *ndiml, f77_integer *ndimr, f77_integer *msub);
//  static inline void laset(const char *uplo, f77_integer CONST_REF m, f77_integer CONST_REF n, double *alpha, double *beta, double *a, f77_integer CONST_REF lda);
//  static inline void lasq1(f77_integer CONST_REF n, double *d__, double *e, double *work, f77_integer *info);
//  static inline void lasq2(f77_integer CONST_REF n, double *z__, f77_integer *info);
//  static inline void lasq3(f77_integer *i0, f77_integer *n0, double *z__, f77_integer *pp, double *dmin__, double *sigma, double *desig, double *qmax, f77_integer *nfail, f77_integer *iter, f77_integer *ndiv, f77_logical *ieee);
//  static inline void lasq4(f77_integer *i0, f77_integer *n0, double *z__, f77_integer *pp, f77_integer *n0in, double *dmin__, double *dmin1, double *dmin2, double *dn, double *dn1, double *dn2, double *tau, f77_integer *ttype);
//  static inline void lasq5(f77_integer *i0, f77_integer *n0, double *z__, f77_integer *pp, double *tau, double *dmin__, double *dmin1, double *dmin2, double *dn, double *dnm1, double *dnm2, f77_logical *ieee);
//  static inline void lasq6(f77_integer *i0, f77_integer *n0, double *z__, f77_integer *pp, double *dmin__, double *dmin1, double *dmin2, double *dn, double *dnm1, double *dnm2);
//  static inline void lasr(const char *side, const char *pivot, const char *direct, f77_integer CONST_REF m, f77_integer CONST_REF n, double *c__, double *s, double *a, f77_integer CONST_REF lda);
//  static inline void lasrt(const char *id, f77_integer CONST_REF n, double *d__, f77_integer *info);
//  static inline void lassq(f77_integer CONST_REF n, double *x, f77_integer CONST_REF incx, double *scale, double *sumsq);
//  static inline void lasv2(double *f, double *g, double *h__, double *ssmin, double *ssmax, double *snr, double *csr, double *snl, double *csl);
//  static inline void laswp(f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *k1, f77_integer *k2, f77_integer *ipiv, f77_integer CONST_REF incx);
//  static inline void lasy2(f77_logical *ltranl, f77_logical *ltranr, f77_integer *isgn, f77_integer *n1, f77_integer *n2, double *tl, f77_integer CONST_REF ldtl, double *tr, f77_integer CONST_REF ldtr, double *b, f77_integer CONST_REF ldb, double *scale, double *x, f77_integer CONST_REF ldx, double *xnorm, f77_integer *info);
//  static inline void lasyf(const char *uplo, f77_integer CONST_REF n, f77_integer *nb, f77_integer *kb, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *w, f77_integer CONST_REF ldw, f77_integer *info);
//  static inline void latbs(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *x, double *scale, double *cnorm, f77_integer *info);
//  static inline void latdf(f77_integer *ijob, f77_integer CONST_REF n, double *z__, f77_integer CONST_REF ldz, double *rhs, double *rdsum, double *rdscal, f77_integer *ipiv, f77_integer *jpiv);
//  static inline void latps(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer CONST_REF n, double *ap, double *x, double *scale, double *cnorm, f77_integer *info);
//  static inline void latrd(const char *uplo, f77_integer CONST_REF n, f77_integer *nb, double *a, f77_integer CONST_REF lda, double *e, double *tau, double *w, f77_integer CONST_REF ldw);
//  static inline void latrs(const char *uplo, const char *trans, const char *diag, const char *normin, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *x, double *scale, double *cnorm, f77_integer *info);
//  static inline void latrz(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *l, double *a, f77_integer CONST_REF lda, double *tau, double *work);
//  static inline void latzm(const char *side, f77_integer CONST_REF m, f77_integer CONST_REF n, double *v, f77_integer CONST_REF incv, double *tau, double *c1, double *c2, f77_integer CONST_REF ldc, double *work);
//  static inline void lauu2(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void lauum(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void opgtr(const char *uplo, f77_integer CONST_REF n, double *ap, double *tau, double *q, f77_integer CONST_REF ldq, double *work, f77_integer *info);
//  static inline void opmtr(const char *side, const char *uplo, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, double *ap, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void org2l(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void org2r(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void orgbr(const char *vect, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orghr(f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orgl2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void orglq(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orgql(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
  static inline void orgqr(f77_integer m, f77_integer n, f77_integer k, double *a, f77_integer lda, double *tau, double *work, f77_integer lwork, f77_integer *info) {
    F77_FUNC(dorgqr)(m, n, k, a, lda, tau, work, lwork, info);

  }
//  static inline void orgr2(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer *info);
//  static inline void orgrq(f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orgtr(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orm2l(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void orm2r(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void ormbr(const char *vect, const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormhr(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer *ilo, f77_integer *ihi, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void orml2(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void ormlq(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormql(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormqr(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormr2(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void ormr3(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer *info);
//  static inline void ormrq(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormrz(const char *side, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void ormtr(const char *side, const char *uplo, const char *trans, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *c__, f77_integer CONST_REF ldc, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void pbcon(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbequ(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *s, double *scond, double *amax, f77_integer *info);
//  static inline void pbrfs(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *afb, f77_integer CONST_REF ldafb, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbstf(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, f77_integer *info);
//  static inline void pbsv(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void pbsvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *afb, f77_integer CONST_REF ldafb, const char *equed, double *s, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void pbtf2(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, f77_integer *info);
//  static inline void pbtrf(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, f77_integer *info);
//  static inline void pbtrs(const char *uplo, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void pocon(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void poequ(f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *s, double *scond, double *amax, f77_integer *info);
//  static inline void porfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *af, f77_integer CONST_REF ldaf, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void posv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void posvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *af, f77_integer CONST_REF ldaf, const char *equed, double *s, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void potf2(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
  static inline void potrf(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info) {
   F77_FUNC(dpotrf)(uplo, n, a, lda, info);
  }
//  static inline void potri(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void potrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ppcon(const char *uplo, f77_integer CONST_REF n, double *ap, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void ppequ(const char *uplo, f77_integer CONST_REF n, double *ap, double *s, double *scond, double *amax, f77_integer *info);
//  static inline void pprfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *afp, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void ppsv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ppsvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *afp, const char *equed, double *s, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void pptrf(const char *uplo, f77_integer CONST_REF n, double *ap, f77_integer *info);
//  static inline void pptri(const char *uplo, f77_integer CONST_REF n, double *ap, f77_integer *info);
//  static inline void pptrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ptcon(f77_integer CONST_REF n, double *d__, double *e, double *anorm, double *rcond, double *work, f77_integer *info);
//  static inline void pteqr(const char *compz, f77_integer CONST_REF n, double *d__, double *e, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void ptrfs(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *d__, double *e, double *df, double *ef, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *info);
//  static inline void ptsv(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *d__, double *e, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ptsvx(const char *fact, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *d__, double *e, double *df, double *ef, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *info);
//  static inline void pttrf(f77_integer CONST_REF n, double *d__, double *e, f77_integer *info);
//  static inline void pttrs(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *d__, double *e, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void ptts2(f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *d__, double *e, double *b, f77_integer CONST_REF ldb);
//  static inline void rscl(f77_integer CONST_REF n, double *sa, double *sx, f77_integer CONST_REF incx);
//  static inline void sbev(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void sbevd(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void sbevx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *q, f77_integer CONST_REF ldq, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sbgst(const char *vect, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, double *ab, f77_integer CONST_REF ldab, double *bb, f77_integer CONST_REF ldbb, double *x, f77_integer CONST_REF ldx, double *work, f77_integer *info);
//  static inline void sbgv(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, double *ab, f77_integer CONST_REF ldab, double *bb, f77_integer CONST_REF ldbb, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void sbgvd(const char *jobz, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, double *ab, f77_integer CONST_REF ldab, double *bb, f77_integer CONST_REF ldbb, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void sbgvx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, f77_integer *ka, f77_integer *kb, double *ab, f77_integer CONST_REF ldab, double *bb, f77_integer CONST_REF ldbb, double *q, f77_integer CONST_REF ldq, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sbtrd(const char *vect, const char *uplo, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *d__, double *e, double *q, f77_integer CONST_REF ldq, double *work, f77_integer *info);
//  static inline void spcon(const char *uplo, f77_integer CONST_REF n, double *ap, f77_integer *ipiv, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void spev(const char *jobz, const char *uplo, f77_integer CONST_REF n, double *ap, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void spevd(const char *jobz, const char *uplo, f77_integer CONST_REF n, double *ap, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void spevx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, double *ap, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void spgst(f77_integer *itype, const char *uplo, f77_integer CONST_REF n, double *ap, double *bp, f77_integer *info);
//  static inline void spgv(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, double *ap, double *bp, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void spgvd(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, double *ap, double *bp, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void spgvx(f77_integer *itype, const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, double *ap, double *bp, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sprfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *afp, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void spsv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void spsvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *afp, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void sptrd(const char *uplo, f77_integer CONST_REF n, double *ap, double *d__, double *e, double *tau, f77_integer *info);
//  static inline void sptrf(const char *uplo, f77_integer CONST_REF n, double *ap, f77_integer *ipiv, f77_integer *info);
//  static inline void sptri(const char *uplo, f77_integer CONST_REF n, double *ap, f77_integer *ipiv, double *work, f77_integer *info);
//  static inline void sptrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void stebz(const char *range, const char *order, f77_integer CONST_REF n, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, double *d__, double *e, f77_integer *m_out, f77_integer *nsplit, double *w, f77_integer *iblock, f77_integer *isplit, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void stedc(const char *compz, f77_integer CONST_REF n, double *d__, double *e, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stegr(const char *jobz, const char *range, f77_integer CONST_REF n, double *d__, double *e, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stein(f77_integer CONST_REF n, double *d__, double *e, f77_integer CONST_REF m, double *w, f77_integer *iblock, f77_integer *isplit, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void steqr(const char *compz, f77_integer CONST_REF n, double *d__, double *e, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void sterf(f77_integer CONST_REF n, double *d__, double *e, f77_integer *info);
//  static inline void stev(const char *jobz, f77_integer CONST_REF n, double *d__, double *e, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *info);
//  static inline void stevd(const char *jobz, f77_integer CONST_REF n, double *d__, double *e, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stevr(const char *jobz, const char *range, f77_integer CONST_REF n, double *d__, double *e, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void stevx(const char *jobz, const char *range, f77_integer CONST_REF n, double *d__, double *e, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sycon(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *anorm, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void syev(const char *jobz, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *w, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void syevd(const char *jobz, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *w, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void syevr(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, f77_integer *isuppz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void syevx(const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void sygs2(f77_integer *itype, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void sygst(f77_integer *itype, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *info);
  static inline void sygv(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *w, double *work, f77_integer CONST_REF lwork, f77_integer *info) {
    F77_FUNC(dsygv)(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);
  }
//  static inline void sygvd(f77_integer *itype, const char *jobz, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *w, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void sygvx(f77_integer *itype, const char *jobz, const char *range, const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *vl, double *vu, f77_integer *il, f77_integer *iu, double *abstol, f77_integer *m_out, double *w, double *z__, f77_integer CONST_REF ldz, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *ifail, f77_integer *info);
//  static inline void syrfs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void sysv(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void sysvx(const char *fact, const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *af, f77_integer CONST_REF ldaf, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *rcond, double *ferr, double *berr, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void sytd2(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *d__, double *e, double *tau, f77_integer *info);
//  static inline void sytf2(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, f77_integer *info);
//  static inline void sytrd(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *d__, double *e, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void sytrf(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void sytri(const char *uplo, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *work, f77_integer *info);
//  static inline void sytrs(const char *uplo, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, f77_integer *ipiv, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void tbcon(const char *norm, const char *uplo, const char *diag, f77_integer CONST_REF n, f77_integer *kd, double *ab, f77_integer CONST_REF ldab, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void tbrfs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void tbtrs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer *kd, f77_integer CONST_REF nrhs, double *ab, f77_integer CONST_REF ldab, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void tgevc(const char *side, const char *howmny, f77_logical *select, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, f77_integer *mm, f77_integer *m_out, double *work, f77_integer *info);
//  static inline void tgex2(f77_logical CONST_REF wantq, f77_logical CONST_REF wantz, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *q, f77_integer CONST_REF ldq, double *z__, f77_integer CONST_REF ldz, f77_integer *j1, f77_integer *n1, f77_integer *n2, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void tgexc(f77_logical CONST_REF wantq, f77_logical CONST_REF wantz, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *q, f77_integer CONST_REF ldq, double *z__, f77_integer CONST_REF ldz, f77_integer *ifst, f77_integer *ilst, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  static inline void tgsen(f77_integer *ijob, f77_logical CONST_REF wantq, f77_logical CONST_REF wantz, f77_logical *select, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *alphar, double *alphai, double *beta, double *q, f77_integer CONST_REF ldq, double *z__, f77_integer CONST_REF ldz, f77_integer *m_out, double *pl, double *pr, double *dif, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void tgsja(const char *jobu, const char *jobv, const char *jobq, f77_integer CONST_REF m, f77_integer CONST_REF p, f77_integer CONST_REF n, f77_integer CONST_REF k, f77_integer *l, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *tola, double *tolb, double *alpha, double *beta, double *u, f77_integer CONST_REF ldu, double *v, f77_integer CONST_REF ldv, double *q, f77_integer CONST_REF ldq, double *work, f77_integer *ncycle, f77_integer *info);
//  static inline void tgsna(const char *job, const char *howmny, f77_logical *select, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, double *s, double *dif, f77_integer *mm, f77_integer *m_out, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void tgsy2(const char *trans, f77_integer *ijob, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *c__, f77_integer CONST_REF ldc, double *d__, f77_integer CONST_REF ldd, double *e, f77_integer CONST_REF lde, double *f, f77_integer CONST_REF ldf, double *scale, double *rdsum, double *rdscal, f77_integer *iwork, f77_integer *pq, f77_integer *info);
//  static inline void tgsyl(const char *trans, f77_integer *ijob, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *c__, f77_integer CONST_REF ldc, double *d__, f77_integer CONST_REF ldd, double *e, f77_integer CONST_REF lde, double *f, f77_integer CONST_REF ldf, double *scale, double *dif, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer *info);
//  static inline void tpcon(const char *norm, const char *uplo, const char *diag, f77_integer CONST_REF n, double *ap, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void tprfs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void tptri(const char *uplo, const char *diag, f77_integer CONST_REF n, double *ap, f77_integer *info);
//  static inline void tptrs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *ap, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void trcon(const char *norm, const char *uplo, const char *diag, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *rcond, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void trevc(const char *side, const char *howmny, f77_logical *select, f77_integer CONST_REF n, double *t, f77_integer CONST_REF ldt, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, f77_integer *mm, f77_integer *m_out, double *work, f77_integer *info);
//  static inline void trexc(const char *compq, f77_integer CONST_REF n, double *t, f77_integer CONST_REF ldt, double *q, f77_integer CONST_REF ldq, f77_integer *ifst, f77_integer *ilst, double *work, f77_integer *info);
//  static inline void trrfs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *x, f77_integer CONST_REF ldx, double *ferr, double *berr, double *work, f77_integer *iwork, f77_integer *info);
//  static inline void trsen(const char *job, const char *compq, f77_logical *select, f77_integer CONST_REF n, double *t, f77_integer CONST_REF ldt, double *q, f77_integer CONST_REF ldq, double *wr, double *wi, f77_integer *m_out, double *s, double *sep, double *work, f77_integer CONST_REF lwork, f77_integer *iwork, f77_integer CONST_REF liwork, f77_integer *info);
//  static inline void trsna(const char *job, const char *howmny, f77_logical *select, f77_integer CONST_REF n, double *t, f77_integer CONST_REF ldt, double *vl, f77_integer CONST_REF ldvl, double *vr, f77_integer CONST_REF ldvr, double *s, double *sep, f77_integer *mm, f77_integer *m_out, double *work, f77_integer CONST_REF ldwork, f77_integer *iwork, f77_integer *info);
//  static inline void trsyl(const char *trana, const char *tranb, f77_integer *isgn, f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, double *c__, f77_integer CONST_REF ldc, double *scale, f77_integer *info);
//  static inline void trti2(const char *uplo, const char *diag, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void trtri(const char *uplo, const char *diag, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, f77_integer *info);
//  static inline void trtrs(const char *uplo, const char *trans, const char *diag, f77_integer CONST_REF n, f77_integer CONST_REF nrhs, double *a, f77_integer CONST_REF lda, double *b, f77_integer CONST_REF ldb, f77_integer *info);
//  static inline void tzrqf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, f77_integer *info);
//  static inline void tzrzf(f77_integer CONST_REF m, f77_integer CONST_REF n, double *a, f77_integer CONST_REF lda, double *tau, double *work, f77_integer CONST_REF lwork, f77_integer *info);
//  f77_integer icmax1(f77_integer CONST_REF n, f77_complex *cx, f77_integer CONST_REF incx);
//  f77_integer ieeeck(f77_integer CONST_REF ispec, f77_real *zero, f77_real *one);
//  f77_integer ilaenv(f77_integer CONST_REF ispec, const char *name__, const char *opts, f77_integer *n1, f77_integer *n2, f77_integer *n3, f77_integer *n4, f77_str_len name_len, f77_str_len opts_len);
//  f77_integer izmax1(f77_integer CONST_REF n, doublecomplex *cx, f77_integer CONST_REF incx);

}; //CppLapack

int CppLapack<float>::getri_block_size=0;
int CppLapack<float>::geqrf_block_size=0;
int CppLapack<float>::orgqr_block_size=0;
int CppLapack<float>::geqrf_dorgqr_block_size=0;

int CppLapack<double>::getri_block_size=0;
int CppLapack<double>::geqrf_block_size=0;
int CppLapack<double>::orgqr_block_size=0;
int CppLapack<double>::geqrf_dorgqr_block_size=0;


static CppLapack<float> cpplapack_float;
static CppLapack<double> cpplapack_double;
}; //la name space

#endif
