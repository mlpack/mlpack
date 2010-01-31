/** @file optimizer.h
 *
 *  @author Nikolaos Vasiloglou (nick)
 *
 * 
 */

#ifndef OPTIMIZATION_CSDP_OPTIMIZER_H_
#define OPTIMIZATION_CSDP_OPTIMIZER_H_

extern "C" {
  #include "declarations.h" 
}
#include "fastlib/fastlib.h"

namespace optim  {
namespace csdp {

struct SparseEntry {
 public:
  index_t row;
  index_t col;
  double  val;
};

typedef ArrayList<SparseEntry> SparseMatrix;
struct SparseBlock {
  index_t       size;
  SparseMatrix  m;
};
typedef ArrayList<SparseBlock> BlockSparseMatrix;
typedef ArrayList<BlockSparseMatrix> BlockSparseMatrixCollection;

class Block {
 public: 
  Block() {
  
  } 
  ~Block() {
  }
  void Init(index_t size, bool is_diagonal) {
    is_diagonal_=is_diagonal;
    size_=size;
    if (is_diagonal_==true) {
      vec_.Init(size);
      mat_.Alias(vec_.ptr(),1, size);      
    } else {
      mat_.Init(size_, size_);
      vec_.Alias(mat_.ptr(), size_);
    }
  }
  void Destruct() {
    vec_.Destruct();
    mat_.Destruct();
  }
  void Copy(const Block &block) {
    is_diagonal_=block.is_diagonal_;
    if (is_diagonal_==true) {
      vec_.Copy(block.vec_);
      mat_.Alias(block.mat_);
    } else {
      mat_.Copy(block.mat_);
      vec_.Alias(block.vec_);
    }
  }
  bool IsDiagonal() {
    return is_diagonal_;
  }
  Matrix &GetMatrix() {
    return mat_;
  }
  Vector &GetVector() {
    return vec_;
  }
  index_t size() {
    if (is_diagonal_==true) {
      return vec_.length();
    } else {
      return mat_.n_rows();
    }
  }

 private:
  bool is_diagonal_;
  Vector vec_;
  Matrix mat_;
};

typedef ArrayList<Block> BlockMatrix;

class CsdpOptimizer {
 public:
  void Init(fx_module *module, 
            BlockMatrix &c, 
            BlockSparseMatrixCollection &a,
            Vector &b) {
    
    dimension_=0;
    num_of_constraints_=a.size();
    module_=module;
    constant_offset_=fx_param_double(module_, "constant_offset", 0.0);
    // Now do a block_size check
    DEBUG_ONLY {
      for(index_t i=0; i<a.size(); i++) {
        if (c.size() !=a[i].size()) {
          FATAL("Matrix %"LI"d has size %"LI"d in C while " "
                it has %"LI"d for A, they must be the same",
                i, c.size(), a[i].size());
        }
        for(index_t j=0; j<c.size(); j++) {
          dimension_+=c[j].size;
          if (c[j].size() != a[i][j].size) {
            FATAL("Block %"LI"d has size %"LI"d "
                  "in C while it has size %"LI"d "
                  "in the %"LI"d matrix of A %"LI"d ",
                  j, c[j].size(), i, a[i][j].size);
          }
        }
      }
    } 
    DEBUG_MSG_IF(b.length()!=a.size(), 
        "B (%"LI"d)must have the same length with  A (%d"LI"d)", 
        b.length(), a.size());
    // We need to do a sanity check on the A matrices to make sure
    // they are symmetric
    DEBUG_ONLY {
   
    }
  }
  void Destruct() {
    free_prob(dimension_, num_of_constraints_, c_, 
              b_, constraints_, x_, y_,z_);
  }
  successs_t Optimize() {
    

    initsoln(dimension_ , num_of_constraints_, 
             c_, b_, constraints_ ,&x_ ,&y_ ,&z_);

    
    // Solve the problem.
    index_t ret=easy_sdp(dimension_, num_of_constraints_,c_,b_,
                         constraints_, constant_offset_, 
                         &x_, &y_, &z_, &pobj_, &dobj_);

    if (ret == 0) {
      NOTIFY("The objective value is %.7e \n",(dobj+pobj)/2);
    } else {
      NOTIFY("SDP failed.\n");
    }
    std::string ret_message;
    success_t result;
    switch(ret) {
      case 1:
        ret_message="Success. The problem is primal infeasible";
        result=SUCCESS_PASS;
        break;
      case 2:
        ret_message="Success. The problem is dual infeasible";
         result=SUCCESS_PASS;
         break;
      case 3:
        ret_message="Partial success. A solution has been found, but full accuracy was not "
            "achieved. One or more of primal infeasibility,"
            "dual infeasibility, or relative "
            "duality gap are larger than their tolerances, "
            "but by a factor of less than  1000";
        result=SUCCESS_PASS;
        break;
       case 4:
         ret_message="Failure. Maximum iterations reached";
         result=SUCCESS_FAIL;
         break;
       case 5:
         ret_message="Failure. Stuck at edge of primal feasibility";
         result=SUCCESS_FAIL;
        break;
       case 6:
        result=SUCCESS_FAIL;
        ret_message="Failure. Stuck at edge of dual infeasibility";
         break;
       case 7:
         result=SUCCESS_FAIL;
         ret_message="Lack of progress";
         break;
       case 8:
         ret_message=" X, Z, or O was singular";
         result=SUCCESS_FAIL;
         break;
       case 9:
         result=SUCCESS_FAIL;
         ret_message="Detected NaN or Inf values";
    }
    fx_result_str("message", ret_message);
    fx_result_double("primal_optimum", pobj_);
    fx_result_double("dual_optimum", dobj_);
    return result;
  }
  void GetX(BlockMatrix *x) {
    ConvertCsdpToBlockMatrixFormat(x_, &x);
  }
  void GetZ(SparseMatrix *z) {
    ConvertCsdpToBlockMatrixFormat(z_, &z);
  }
  void GetY(Vector *y) {
    y->Copy(y_, num_of_constraints_);
  }
  void GetOptimum(double *primal_optimal, 
                  double *dual_optimal) {
    *primal_optimal=pobj_;
    *dual_optimal = dbj_;
  }
 
 private:  
  fx_module *module_;
  index_t dimension_;
  index_t num_of_constraints_;
  double constant_offset_;
  struct blockmatrix c_;
  struct constraintmatrix *constraints_;
  double *b_;
  // Storage for the initial and final solutions.
  struct blockmatrix x,z;
  double *y;
  double pobj,_dobj_;

   
  void ConvertSparseMatrixToCsdpFormat(BlockMatrix &c,
                                       BlockSparseMatrixCollection &a,
                                       Vector &b) {
   struct sparseblock *blockptr;
    c_.nblocks=c.size();
    c_.blocks=mem::Alloc<struct blockrec>(c.size()+1);
    if (c_.blocks == NULL) {
        FATAL("Couldn't allocate storage for C!\n");
    };
    for(index_t i=0; i<c.size(); i++) {
      if (c[i].IsDiagonal()==false) {
        c_.blocks[i+1].blockcategory=MATRIX;
        c_.blocks[i+1].blocksize=c[i].size()+1;
        c_.blocks[i+1].data.mat=mem::Alloc<double>(math::Sqr(c[i].size()));
        if (C.blocks[i+1].data.mat == NULL) {
          FATAL("Couldn't allocate storage for C!\n");
        };
        for(index_t j=0; j<c[i].size(); j++) {
          for(index_t k=0; k<c[i].size(); k++) {
            c_.blocks[i+1].data.mat[ijtok(j+1, k+1,i+2)]
              =c[i].GetMatrix().get(j, k);
          } 
        }
      } else {
        c_.blocks[i+1].blockcategory=DIAG;
        c_.blocks[i+1].blocksize=c[i].size()+1;
        c_.blocks[i+1].data.vec=mem::Alloc<double>(c[i].size());
        if (C.blocks[i+1].data.vec == NULL) {
          FATAL("Couldn't allocate storage for C!\n");
        };
        for(index_t j=0; j<c[i].size(); j++) {
          c_.blocks[i+1].data.vec[j+1]=c[i].GetVector()[j];
        }
      }
    }
    constraints_=mem::Alloc<struct constraintmatrix>(a.size()+1);
    if (constraints==NULL) {
      FATAL("Failed to allocate storage for constraints!\n");
    };

    for(index_t i=0; i<=a.size(); i++) {
      constraints_[i+1].blocks=NULL;
      for(index_t j=a[i].size()-1; j>=0; j--) {
        blockptr = mem::Alloc<sparseblock>();
        if (blockptr==NULL) {
          FATAL("Allocation of constraint block failed!\n");
        };
        blockptr->blocknum=j+1;
        blockptr->blocksize=a[i][j].size+1;
        blockptr->constraintnum=i+1;
        blockptr->next=NULL;
        blockptr->nextbyblock=NULL;
        blockptr->entries= mem::Alloc<double> malloc(a[i][j].size+1);
        if (blockptr->entries==NULL) {
          FATAL("Allocation of constraint block failed!\n");
        };
        blockptr->iindices=mem::Alloc<int>(a[i][j].size+1);
        if (blockptr->iindices==NULL) {
          FATAL("Allocation of constraint block failed!\n");
        };
        blockptr->jindices=mem::Alloc<int>(a[i][j].size+1);
        if (blockptr->jindices==NULL) {
          FATAL("Allocation of constraint block failed!\n");
        };

        blockptr->numentries=a[i][j].m.size();
        for(index_t k=0; k< a[i][j].m.size(); k++) {
          blockptr->iindices[k+1]=a[i][j].m[k].row+1;
          blockptr->jindices[k+1]=a[i][j].m[k].col+1;
          blockptr->entries[k+1]=a[i][j].m[k].val;
        }
        blockptr->next=constraints_[i+1].blocks;
        constraints_[i+1].blocks=blockptr;
      }
    }
    b_=mem::Alloc<double>(b.length());
    memcpy(b_, b.ptr(), b.length()*sizeof(double));
  }

  void ConvertCsdpToBlockMatrixFormat(struct blockmatrix &csdp_mat,
                                       BlockMatrix *new_mat ) {
    new_mat->Init(csdp_mat.nblocks);
    for(index_t i=0; i<csdp_mat.nblocks; i++) {
      if (blocks[i+1].blockcategory==MATRIX) {
        (*new_mat)[i].Init(false, csdp_mat[i+1].blocksize);
        for(index_t j=0; j<csdp_mat[i].blocksize; j++) {
          for(index_t k=0; k<csdp_mat[i].blocksize; k++) { 
            (*new_mat)[i].GetMatrix().set(j, k, 
                csdp_mat_.blocks[i+1].data.mat[ijtok(j+1, k+1,i+2)]);
          }
        }
      } else {
        (*new_mat)[i].Init(true, csdp_mat[i+1].blocksize);   
        for(index_t j=0; j<csdp_mat[i].blocksize; j++) {
          (*new_mat)[i].GetVector().[j]=
                csdp_mat_.blocks[i+1].data.mat[j+1];
        }
      }
    } 
  }
};     
  
};// namespace optim
};// namespace csdp
# endif
