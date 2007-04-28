

skeleton


   f <- a + d                                                             
   r1 <- a + b                                                             
   r2 <- c + d                                                             
   r3 <- r1 + r2                                                             
   a <- b                                                             
   g <- q                                                             
   x <- f + g                                                             
 + f <- x + c                                                             
   live-out: x, f                                                             

  e = [h * i]
  c = [f * g]
  x = [a, b, c, d, e]
  
  r1 <- a + b
  r2 <- r1 + b
  x = ?

bool r_is_add(Opcode opc) {
  return opc == ADD_W || opc == ADDL_W;
}

bool r_is_sub(Opcode opc) {
  return opc == SUB_W || opc == SUBL_W;
}

bool r_is_addsub(Opcode opc) {
  return r_is_add(opc) || r_is_sub(opc);
}

bool r_is_mpy(Opcode opc) {
  return opc == MPY_W || opc == MPYL_W;
}

bool r_can_reassoc(Opcode opc) {
  return r_is_addsub(opc) || r_is_mpy(opc);
}

struct Assoc_add_entry {
  int sign;
  int rank;
  const Operand& operand;
  
  Assoc_add_entry(int sign_in, int rank_in, const Operand& operand_in)
      : sign(sign_in)
      , rank(rank_in)
      , operand(operand_in) {
  }
  
  friend operator < (const Assoc_add_entry& a, const Assoc_add_entry& b) {
    if (a.sign != b.sign) {
      return a.sign > b.sign;
    } else {
      return a.rank < b.rank;
    }
  }

  friend operator == (const Assoc_add_entry& a, const Assoc_add_entry& b) {
    return (a.sign == b.sign) && (a.rank == b.rank);
  }
};

struct Assoc_mul_entry {
  int rank;
  const Operand& operand;
  
  Assoc_add_entry(iint rank_in, const Operand& operand_in)
      : rank(rank_in)
      , operand(operand_in) {
  }

  friend operator < (const Assoc_mul_entry& a, const Assoc_mul_entry& b) {
    return a.rank < b.rank;
  }

  friend operator == (const Assoc_mul_entry& a, const Assoc_mul_entry& b) {
    return (a.rank == b.rank);
  }
};

#define NO_RANK 0

class Associator {
 public:
  
  void Combine_adds(int sign, const Operand& operand,
      Slist<Assoc_add_entry> *dest) const {
    if (adds.is_bound(operand)) {
      for (Slist_iterator<Assoc_add_entry> it(adds.value(operand)); it != 0; ++it) {
        dest->add(Assoc_add_entry(it->sign * sign, it->rank, it->operand));
      }
    } else {
      dest->add(Assoc_add_entry(sign, NO_RANK, operand));
    }
  }
  
  void Combine_muls(const Operand& operand,
      Slist<Assoc_mul_entry> *dest) const {
    if (adds.is_bound(operand)) {
      for (Slist_iterator<Assoc_mul_entry> it(muls.value(operand)); it != 0; ++it) {
        dest->add(Assoc_mul_entry(it->rank, it->operand));
      }
    } else {
      dest->add(Assoc_mul_entry(NO_RANK, operand));
    }
  }
  
 private:
  Basicblock *bb;
  Hash_set<int> delete_set;
  Map<Operand, Slist<Assoc_add_entry>> adds;
  Map<Operand, Slist<Assoc_mul_entry>> muls;
};

/**
 * Initial backwards pass that finds out which lines must be computed
 * in-place.
 */
void Associator::find_delete_set() {
  Hash_set<Operand> mandatory_live;
  Hash_map<Operand, int> first_def;
  Hash_map<Operand, int> last_use;
  int max_pos = 0; /* 0 sounds good, everything can be negative... */
  int pos = max_pos;
  
  mandatory_live = live_out;
  
  for (Hash_set<Operand>::iterator it(mandatory_live); it != 0; ++it) {
    last_use.bind(*it, pos + 1);
  }
  
  for (Region_ops_linear op_iter(bb, true); op_iter != 0; --op_iter, --pos) {
    Op* cur_op = *op_iter;
    bool can_reassoc = r_can_reassoc(op);
    bool must_compute = !can_reassoc;
    int my_dests_last_use = -999999; /* used before the block starts */
    int my_inputs_first_redef = max_pos;
    
    for (Op_complete_dests dest_oper(dest_oper); dest_oper != 0; ++dest_oper) {
      if (dest_oper->is_reg()) {
        if (mandatory_live.contains(*dest_oper)) {
          must_compute = true;
        }
        mandatory_live -= *dest_oper;
        
        if (last_use.is_bound(*dest_oper)) {
          my_dests_last_use = max(my_dests_last_use, last_use.value(*dest_oper));
        }
        
        last_use.unbind(*dest_oper);
        first_def.bind(*dest_oper, pos);
      }
    }
    
    for (Op_complete_inputs input_oper(cur_op); input_oper != 0; ++input_oper) {
      if (input_oper->is_reg()) {
        if (!can_reassoc) {
          mandatory_live += *input_oper;
        }
        
        if (!last_use.is_bound(*input_oper)) {
          last_use.bind(*input_oper, pos);
        }
        
        if (first_def.is_bound(*input_oper)) {
          my_inputs_first_redef = min(my_inputs_first_redef, first_def.value(*input_oper));
        }
      }
    }
    
    if (my_inputs_first_redef <= my_dests_last_use) {
      // We must compute it if one of my inputs are redefined
      // before I am last used.
      must_compute = true;
    }
    
    if (!must_compute) {
      delete_set += cur_op->id();
    }
  }
}

/**
 * Abstraction to automatically handle whether it's necessary
 * to generate a temporary.
 *
 * This class encapsulates either a computation, or an operand.
 * It has two abilities: to assign to an existing operand, or to
 * generate an operand.
 *
 * If this class encapsulates an operand, then making an operand
 * just returns the original, but assignment requires a move.
 * If this encapsulates an operation, then making an operand
 * assigns the dest to a temp register, but assignment just
 * requires changing the destination of the operation.
 */
class Result {
 private:
  Op* op_;
  const Operand* oper_;
  Data_type data_type_;
  bool is_const_;
  int val_;
  
 public:
  Result() : op_(NULL), oper_(NULL), val_(-1) {}

  Result(int constant, const Data_type& type)
      : op_(NULL), oper_(new Int_lit(constant, type)), data_type_(type),
        is_const_(true), val_(constant) {}
  
  Result(Op* op_in, Datatype datatype_in)
      : op_(op_in), oper_(NULL), data_type_(datatype_in), is_const_(false),
        val_(-1) {}
  
  Result(const Operand* oper_in)
      : op_(NULL), oper_(oper_in), data_type_(oper_in->data_type()) {
    if (oper_in->is_lit() && oper_in->is_int()) {
      is_const_ = true;
      val_ = oper_in->int_value();
    } elsse {
      is_const_ = false;
      val_ = -1;
    }
  }
  
  const Data_type& data_type() {
    return data_type_;
  }
  
  Op* assign_to(const Operand& dest, Op *successor) {
    if (op_) {
      op_->set_dest(DEST1, dest);
    } else if (oper_) {
      op_ = new Op(get_move_opcode_for_operand(dest));
      
      Operand pred_true(new Pred_lit(true));
      new_op->set_src(PRED1, pred_true);

      Op_explicit_sources sources(new_op);
      *sources = *oper_;
      
      El_insert_op_before(successor->parent(), new_op, successor);
    }
    return op_;
  }
  
  /**
   * Returns a temporary or permanent operand.
   */
  Operand make_operand() {
    Operand value;
    if (op_) {
      value = Reg(data_type_);
      new_op->set_dest(DEST1, value);
    } else if (oper_) {
      value = *oper_;
    }
    return value;
  }
  
  bool is_zero() const {
    return is_const() && val_ == 0;
  }
  
  bool is_one() const {
    return is_const() && val_ == 1;
  }
  
  bool is_const() const {
    return is_const_;
  }
  
  int val() const {
    return val_;
  }
};

Result Associator::insert_binop_before(const Result& lhs, const Result& rhs,
    const Opcode& opcode, Op *successor) {
  bool is_add = r_is_add(opcode);
  bool is_sub = r_is_sub(opcode);
  bool is_mul = r_is_mul(opcode);

  if (is_add() && lhs.is_zero()) {
    return Result(rhs);
  } else if (is_add && rhs.is_zero()) {
    return Result(lhs);
  } else if (is_sub && rhs.is_zero()) {
    return Result(lhs);
  } else if (is_mul && (lhs.is_zero() || rhs.is_zero())) {
    return Result(0, lhs.data_type());
  } else if (lhs.is_const() && rhs.is_const()) {
    if (is_add) {
      return Result(lhs.val() + rhs.val(), lhs.data_type());
    } else if (is_sub) {
      return Result(lhs.val() - rhs.val(), lhs.data_type());
    } else if (is_mul) {
      return Result(lhs.val() * rhs.val(), lhs.data_type());
    }
  }

  Op *new_op = new Op(opcode);

  new_op->set_src(PRED1, Operand(new Pred_list(true)));
  new_op->set_src(SRC1, lhs.make_operand());
  new_op->set_src(SRC2, rhs.make_operand());

  assert(bb == successor->parent());

  El_insert_op_before(successor->parent(), new_op, successor);

  ASSERT(lhs.data_type() == rhs.data_type());

  return Result(new_op, lhs.data_type());
}

// THIS IS BROKEN!!!!  AHH!!!
Result Associator::gen_code(const Operand& name, Op *successor) {
  Op* last_op = NULL;
  const Operand* last_operand = NULL;
  Operand dest;
  Result result;

  if (adds.is_bound(name)) {
    Slist<Assoc_add_entry> adds;
    Slist<Assoc_add_entry> subs;
    Opcode add_opcode = get_add_opcode_for_operand(name);
    Opcode sub_opcode = get_sub_opcode_for_operand(name);

    for (Slist_iterator<Assoc_add_entry> all_it(adds.value(name));
         all_it != 0; ++all_it) {
      if (all_it->sign < 0) {
        subs.add(*all_it);
      } else {
        adds.add(*all_it);
      }
    }

    Slist_iterator<Assoc_add_entry> add_iterator(adds);
    Slist_iterator<Assoc_add_entry> sub_iterator(subs);

    result = Result(0, name.data_type());

    for (; sub_iterator != 0; ++sub_iterator) {
      Result subresult = gen_code(sub_iterator->operand, successor);
      result = insert_binop_before(result, subresult,
          add_opcode, successor);
    }

    Result subresult;

    if (add_iterator != 0) {
      gen_code(add_iterator->operand, succesor);
      insert_binop_before(subresult, result, sub_opcode, successor);
      ++add_iterator;

      for (; add_iterator != 0; ++add_iterator) {
        Result subresult = gen_code(add_iterator->operand, successor);
        result = insert_binop_before(result, subresult, add_opcode, successor);
      }      
    } else {
      insert_binop_before(Result(0, name.data_type()), result,
          sub_oppcode, successor);
    }
  } else if (muls.is_bound(name)) {
    Slist_iterator<Assoc_mul_entry> entry_it(adds.value(name));
    Opcode opcode = get_mpy_opcode_for_operand(name);

    result = Result(1, name.data_type());

    for (; entry_it != 0; ++entry_it) {
      Result subresult = gen_code(entry_it->operand, successor);
      result = insert_binop_before(result, subresult, opcode, successor);
    }
  } else {
    result = Result(&name);
  }

  return result;
}

void Associator::fix_basic_block() {
  for (Region_ops_linear op_iter(bb, false); op_iter != 0; ++op_iter) {
    Op* cur_op = *op_iter;
    bool can_reassoc = CAN_REASSOC(op);

    if (can_reassoc) {
      Op_explicit_inputs input_oper(cur_op);
      Operand& op1 = *input_oper;
      Operand& op2 = *++input_oper;
      Op_explicit_dests dest_oper_it(cur_op);
      Operand& dest_oper = *dest_oper_it;
      Opcode opcode = op->opcode();

      if (r_is_addsub(opcode)) {
        Slist<Assoc_add_entry> mylist;
        int sign = r_is_add(opcode) ? 1 : -1;

        Combine_adds(1, op1, &mylist);
        Combine_adds(sign, op2, &mylist);

        adds.bind(dest_oper, mylist);
      } else if (r_is_mpy(opcode)) {
        Slist<Assoc_mul_entry> mylist;

        Combine_muls(op1, &mylist);
        Combine_muls(op2, &mylist);

        muls.bind(dest_oper, mylist);
        opcodes_bind(dest_oper, opcode);
      } else {
        abort();
      }
    }
    
    if (!delete_set.contains(cur_op->id())) {
      if (can_reassoc) {
        const Operand& dest_oper = cur_op->dest(cur_op->first_dest());
        Result result = gen_code(dest_oper, cur_op);
        result.assign_to(dest_oper, cur_op);
        // remove from the sets -- make sure this is referenced explicitly
        adds.unbind(dest_oper);
        muls.unbind(dest_oper);
      } else {
        for (Op_complete_inputs input_oper(cur_op); input_oper != 0; ++input_oper) {
          if (input_oper->is_reg()) {
            Result result = gen_code(dest_oper, cur_op);
            *input_oper = result.make_operand();
          }
        }
      }
    }
  }
  
  for (Hash_set_iterator iter(delete_set); iter != 0; ++iter) {
    Op* op = (Op*)graph.b_map[iter];
    El_remove_op(op);
  }
}
