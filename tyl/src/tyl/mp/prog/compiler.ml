
(* pre: isDLF e *)
let compileDLF e = match e with 
  | EVar _ -> e (*already real (?)*)
  | EConst (Bool true) -> EConst (Real 1.0)
  | EConst (Bool false) -> EConst (Real 0.0)
  | EUnaryOp (Not,e') -> EBinaryOp (Minus,EConst(Real 1.0),compileDLF e')
  | EBinaryOp (Or,e1,e2) -> EBinaryOp (Plus,compileDLF e1,compileDLF e2)
