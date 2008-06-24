open Ast
open List
open Vars

module C = Ctxt
module S = Util.Id.Set

(* The following functions assume 'pre: e is boolean' *)

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
    let (&.) e1 e2 = EBinaryOp(And,e1,e2) in
    let (|.) e1 e2 = EBinaryOp(Or,e1,e2) in
    let nt e1 = EUnaryOp(Not,e1) in
      match e' with 
        | EUnaryOp (Not,EUnaryOp(Not,e1))          -> toCNF e1
        | EUnaryOp (Not,EBinaryOp(Or,e1,e2))       -> toCNF (nt e1 &. nt e2)
        | EUnaryOp (Not,EBinaryOp(And,e1,e2))      -> toCNF (nt e1 |. nt e2)
        | EBinaryOp (Or,EBinaryOp(And,e11,e12),e2) -> toCNF ((e11 |. e2) &. (e12 |. e2))
        | EBinaryOp (Or,e1,EBinaryOp(And,e21,e22)) -> toCNF ((e21 |. e1) &. (e22 |. e1))
        | EBinaryOp (Or,e1,e2)                     -> toCNF (toCNF e1 |. toCNF e2)
        | EBinaryOp (And,e1,e2)                    -> toCNF (toCNF e1 &. toCNF e2)
        | _ -> failwith "should be boolean expressions only"
  in
    if isCNF e then e else toCNF' e

(* *)

let bounded t = match t with 
  | TReal (IntInterval (Some _, Some _)) -> true
  | TReal (RealInterval (Some _, Some _)) -> true
  | TBool _ -> true
  | _ -> false

let rec existVarsBounded c = match c with 
  | CBoolVal _ 
  | CIsTrue _ 
  | CNumRel _              -> true
  | CPropOp (_,cs)         -> for_all existVarsBounded cs
  | CQuant (Exists,_,t,c') -> bounded t && existVarsBounded c'

let rec disjVarsBounded ctxt c = match c with
  | CBoolVal _
  | CIsTrue _
  | CNumRel _              -> true
  | CPropOp (Disj,cs)      -> 
      let freeVars = S.union' (map freeVarsc cs) in
      let ts = map (C.lookup ctxt) (S.elements freeVars) in
        for_all bounded ts && for_all existVarsBounded cs 
  | CPropOp (Conj,cs)      -> for_all (disjVarsBounded ctxt) cs 
  | CQuant (Exists,x,t,c') -> disjVarsBounded (C.add ctxt x t) c'


(* Compilation *)

let compileType t = match t with 
  | TBool None -> TReal (IntInterval (Some 0, Some 1))
  | TBool (Some a) -> 
      let b = Some (if a then 1 else 0) in 
        TReal (IntInterval (b,b))
  | _ -> t (* numeric types are unchanged *)

let rec compileDLF e = assert (isDLF e) ;
  match e with 
    | EVar _ -> e 
    | EConst (Bool true) -> EConst (Real 1.0)
    | EConst (Bool false) -> EConst (Real 0.0)
    | EUnaryOp (Not,e') -> EBinaryOp (Minus,EConst(Real 1.0),compileDLF e')
    | EBinaryOp (Or,e1,e2) -> EBinaryOp (Plus,compileDLF e1,compileDLF e2)
    | _ -> assert false

let rec compileProp ctxt c = assert (disjVarsBounded ctxt c) ;
  match c with 
    | CBoolVal _ -> c
    | CIsTrue e -> 
        let e' = toCNF e in 
          if isDLF e'
          then CNumRel (Gte,compileDLF e',EConst (Real 1.0))
          else compileConj e'
    | CNumRel _ -> c
    | CPropOp (Disj,_) -> compileDisj c
    | CPropOp (Conj,cs) -> CPropOp (Conj, map (compileProp ctxt) cs)
    | CQuant (Exists,x,t,c) -> 
        let t' = compileType t in
        let c' = compileProp (C.add ctxt x t') c in
          CQuant (Exists,x,t',c')

and compileConj e = assert (isConj e) ; 
  match e with 
    | EBinaryOp (And,e1,e2) -> 
        let c1 = compileProp (C.fromList []) (CIsTrue e1) in
        let c2 = compileProp (C.fromList []) (CIsTrue e2) in 
          CPropOp (Conj,[c1;c2]) 
    | _ -> assert false

and compileDisj c = c

let compile p = 
(*   assert (isMP p && disjVarsBounded p) *)
  match p with 
    | PMain (d,xts,e,c) -> 
        let xts' = map (fun (x,t) -> (x,compileType t)) xts in 
        let c' = compileProp (C.fromList xts) c in
          PMain (d,xts',e,c')
