
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

bool r_get_sign(Opcode opc) {
  return (r_is_sub(opc)) ? -1 : 1;
}

bool r_can_reassoc(Opcode opc) {
  return r_is_addsub(opc) || r_is_mpy(opc);
}

Opcode r_get_canonical(Opcode opc) {
  if (opc == ADD_W || opc == SUB_W) {
    return ADD_W;
  } else if (opc == ADDL_W || opc == SUBL_W) {
    return ADDL_W;
  } else if (opc == MPY_W) {
    return MPY_W;
  } else if (opc == MPYL_W) {
    return MPYL_W;
  }
}

struct Assoc_op_entry {
  int sign;
  int rank;
  const Operand& operand;
  
  Assoc_op_entry(int sign_in, int rank_in, const Operand& operand_in)
      : sign(sign_in)
      , rank(rank_in)
      , operand(operand_in) {
  }
  
  friend operator < (const Assoc_op_entry& a, const Assoc_op_entry& b) {
    if (a.rank != b.rank) {
      return a.rank < b.rank;
    } else {
      return a.sign > b.sign;
    }
  }

  friend operator == (const Assoc_op_entry& a, const Assoc_op_entry& b) {
    return (a.sign == b.sign) && (a.rank == b.rank);
  }
};

#warning "Rank information not used"
#define NO_RANK 0

class Associator {
 public:
  void combine_ops(int sign, Opcode canonical,
      const Operand& operand,
      Slist<Assoc_op_entry> *dest) const;

 private:
  Basicblock *bb;
  Hash_set<int> delete_set;
  Map<Operand, Slist<Assoc_op_entry>> adds;
  Map<Operand, Opcode> opcodes;
};

void Associator::combine_ops(
    int sign, Opcode canonical,
    const Operand& operand,
    Slist<Assoc_op_entry> *dest) const {
  if (opcodes.is_bound(operand) && opcodes.value(operand) == canonical) {
    for (Slist_iterator<Assoc_op_entry> it(adds.value(operand));
        it != 0; ++it) {
      dest->add(Assoc_op_entry(it->sign * sign, it->rank, it->operand));
    }
  } else {
    dest->add(Assoc_op_entry(sign, NO_RANK, operand));
  }
}


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
    bool can_reassoc = false;
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
  
  Result(Op* op_in, Datatype data_type_in)
      : op_(op_in)
      , oper_(NULL)
      , data_type_(data_type_in)
      , is_const_(false)
      , val_(-1)
  {}
  
  Result(const Operand* oper_in)
      : op_(NULL)
      , oper_(oper_in)
      , data_type_(oper_in->data_type())
  {
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
      op_->set_src(PRED1, Operand(new Pred_lit(true)));
      op_->set_src(SRC1, *oper_);
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

  assert(lhs.data_type() == rhs.data_type());
  return Result(new_op, lhs.data_type());
}

// THIS IS BROKEN!!!!  AHH!!!
Result Associator::gen_code(const Operand& name, Op *successor) {
  Op* last_op = NULL;
  const Operand* last_operand = NULL;
  Operand dest;
  Result result;

  if (opcodes.is_bound(name)) {
    Opcode opcode = opcodes.value(name); // the canonical opcode
    Opcode add_opcode;
    Opcode sub_opcode;

    if (r_is_mpy(opcode)) {
      result = Result(new Int_lit(1, name.data_type()));
      add_opcode = get_mpy_opcode_for_operand(name);
    } else if (r_is_add(opcode)) {
      result = Result(new Int_lit(0, name.data_type()));
      add_opcode = get_add_opcode_for_operand(name);
      sub_opcode = get_sub_opcode_for_operand(name);
    }

    for (Slist_iterator<Assoc_op_entry> subexpr(adds.value(name));
         subexpr != 0; ++subexpr) {
      Result subresult = gen_code(subexpr->operand, successor);
      result = insert_binop_before(result, subresult,
          subexpr->sign == 1 ? add_opcode : sub_opcode, successor);
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
    bool must_compute = !delete_set.contains(cur_op->id());

    if (can_reassoc) {
      Operand lhs = cur_op->src(SRC1);
      Operand rhs = cur_op->src(SRC2);
      Operand dest = cur_op->dest(DEST1);
      Slist<Assoc_op_entry> mylist;
      Opcode opcode = get_root(cur_op->opcode());
      Opcode canonical = r_get_canonical(opcode);

      combine_ops(1, canonical, op1, &mylist);
      combine_ops(r_get_sign(opcode), canonical, op2, &mylist);

      adds.bind(dest, mylist);
      opcodes.bind(dest, opcode);

      if (must_compute) {
        gen_code(dest_oper, cur_op).assign_to(dest, cur_op);
        // this must be referenced by name and not recalculated, so remove
        // from sets.
        opcodes.unbind(dest_oper);
        adds.unbind(dest_oper);
      }
    } else if (must_compute) {
      for (Op_complete_inputs input_oper(cur_op);
          input_oper != 0; ++input_oper) {
        *input_oper = gen_code(*input_oper, cur_op).make_operand();
      }
    }
  }

  for (Hash_set_iterator iter(delete_set); iter != 0; ++iter) {
    Op* op = (Op*)graph.b_map[iter];
    El_remove_op(op);
  }
}
