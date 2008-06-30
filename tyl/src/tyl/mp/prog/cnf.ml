open Ast

module E = Edsl

(* The following functions assume e consists of only boolean syntax *)

let rec isLiteral e = match e with
  | EVar _            -> true
  | EConst (Bool _)   -> true
  | EUnaryOp (Not,e') -> isLiteral e'
  | _ -> false

let rec isDLF e = isLiteral e || match e with 
  | EBinaryOp (Or,e1,e2) -> isDLF e1 && isDLF e2
  | _ -> false

let rec isCNF e = isDLF e || match e with 
  | EBinaryOp (And,e1,e2) -> isCNF e1 && isCNF e2
  | _ -> false

let isConj e = isCNF e && not (isDLF e)

let rec toCNF e = 
  let rec toCNF' e' =     
    match e' with 
      | EUnaryOp (Not,EUnaryOp(Not,e1))          -> toCNF e1
      | EUnaryOp (Not,EBinaryOp(Or,e1,e2))       -> toCNF (E.(&&) (E.not e1) (E.not e2))
      | EUnaryOp (Not,EBinaryOp(And,e1,e2))      -> toCNF (E.(||) (E.not e1) (E.not e2))
      | EBinaryOp (Or,EBinaryOp(And,e11,e12),e2) -> toCNF (E.(&&) (E.(||) e11 e2) (E.(||) e12 e2))
      | EBinaryOp (Or,e1,EBinaryOp(And,e21,e22)) -> toCNF (E.(&&) (E.(||) e21 e1) (E.(||) e22 e1))
      | EBinaryOp (Or,e1,e2)                     -> toCNF (E.(||) (toCNF e1) (toCNF e2))
      | EBinaryOp (And,e1,e2)                    -> toCNF (E.(&&) (toCNF e1) (toCNF e2))
      | _ -> failwith "toCNF: expression contains non-boolean syntax"
  in
    if isCNF e then e else toCNF' e

