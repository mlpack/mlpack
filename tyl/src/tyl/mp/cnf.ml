open Ast
open Util
open Vars

module E = Edsl

(* The CNF functions assume e consists of only boolean syntax *)

let rec isLiteral e = match e with EVar _ -> true | EConst (Bool _) -> true | EUnaryOp (Not,e') -> isLiteral e' | _ -> false
let rec isDLF e = isLiteral e || match e with EBinaryOp (Or,e1,e2) -> isDLF e1 && isDLF e2 | _ -> false
let rec isCNF e = isDLF e || match e with EBinaryOp (And,e1,e2) -> isCNF e1 && isCNF e2 | _ -> false
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
      | _                                        -> assert false
  in
    if isCNF e then e else toCNF' e

let rec containsQuantifier c = match c with
  | CQuant _ -> true
  | CPropOp (_,cs) -> any containsQuantifier cs
  | _ -> false
      
let rec isPNF c = match c with
  | CIsTrue _ -> true
  | CBoolVal _ -> true
  | CNumRel _ -> true
  | CQuant (_,_,_,c') -> isPNF c'
  | CPropOp (_,cs) -> not $ any containsQuantifier cs

let rec toPNF c = match c with 
  | CIsTrue _ -> assert false
  | CBoolVal _ -> c
  | CNumRel _ -> c
  | CQuant (Exists,x,t,c') -> CQuant (Exists,x,t,toPNF c')
  | CPropOp (op,cs) -> 
      let rec mergePNF op c1 c2 = (* pre: c1 and c2 are PNF *)
        match c1,c2 with
          | CQuant(Exists,x,t,c1'), _ -> CQuant(Exists,x,t,mergePNF op c1' c2)
          | _, CQuant(Exists,x,t,c2') -> 
              let (x',c2'') = alphaConvertc x c2' (freeVarsc c1) 
              in CQuant(Exists,x',t,mergePNF op c1 c2'')
          | _,_                       -> CPropOp(op,[c1;c2])
      in 
        foldl1 (mergePNF op) (map toPNF cs)
