open Ast
open List
open Vars
open Wf

module C = Context
module S = Util.Id.Set
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

(* *)

let bounded t = assert (isType t) ; 
  match t with 
  | TReal (Discrete (Some _, Some _))
  | TReal (Continuous (Some _, Some _))
  | TBool _ -> true
  | _       -> false

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
      let xs = S.elements (S.union' (map freeVarsc cs)) in
      let ts = map (C.lookup ctxt) xs in
        for_all bounded ts && for_all existVarsBounded cs 
  | CPropOp (Conj,cs)      -> for_all (disjVarsBounded ctxt) cs 
  | CQuant (Exists,x,t,c') -> disjVarsBounded (C.add ctxt x t) c'


(* Compilation *)

let compileType t = match t with 
  | TBool None -> E.discrete 0 1
  | TBool (Some a) -> let b = if a then 1 else 0 in E.discrete b b 
  | _ -> t (* numeric types are left unchanged *)

let rec compileDLF e = 
  assert (isDLF e) ;
  match e with 
    | EVar _ -> e 
    | EConst (Bool true) -> E.one
    | EConst (Bool false) -> E.zero
    | EUnaryOp (Not,e') -> E.(-) E.one (compileDLF e')
    | EBinaryOp (Or,e1,e2) -> E.(+) (compileDLF e1) (compileDLF e2)
    | _ -> assert false

let rec compileProp ctxt c = 
  assert (disjVarsBounded ctxt c) ;
  match c with 
    | CBoolVal _ -> c
    | CIsTrue e -> let e' = toCNF e in 
        if isDLF e' then E.(>=) (compileDLF e') E.one else compileConj e'
    | CNumRel _ -> c
    | CPropOp (Disj,_) -> compileDisj c
    | CPropOp (Conj,cs) -> CPropOp (Conj, map (compileProp ctxt) cs)
    | CQuant (Exists,x,t,c) -> 
        let t' = compileType t in
        let c' = compileProp (C.add ctxt x t') c in
          CQuant (Exists,x,t',c')

and compileConj e = 
  assert (isConj e) ; 
  match e with 
    | EBinaryOp (And,e1,e2) -> 
        let c1 = compileProp C.empty (E.isTrue e1) in
        let c2 = compileProp C.empty (E.isTrue e2) in 
          E.(/|) c1 c2
    | _ -> assert false

and typeAsProp (x',t) = let x = EVar x' in
  match t with
    | TReal (Continuous (Some lo, Some hi)) -> E.(/|) (E.(<=) (E.litR lo) x) (E.(<=) x (E.litR hi))
    | TReal (Continuous (Some lo, None   )) ->        (E.(<=) (E.litR lo) x)              
    | TReal (Continuous (None   , Some hi)) ->                               (E.(<=) x (E.litR hi))
    | TReal (Continuous (None   , None   )) -> E.propT
    | TReal (Discrete   (Some lo, Some hi)) -> E.(/|) (E.(<=) (E.litI lo) x) (E.(<=) x (E.litI hi))
    | TReal (Discrete   (Some lo, None   )) ->        (E.(<=) (E.litI lo) x)              
    | TReal (Discrete   (None   , Some hi)) ->                               (E.(<=) x (E.litI hi))
    | TReal (Discrete   (None   , None   )) -> E.propT
    | TBool (Some true)  -> E.isTrue x
    | TBool (Some false) -> E.isTrue (E.not x)
    | TBool None         -> E.propT

and addBoundsToProp ctxt c = 
  let xs = S.elements (freeVarsc c) in 
  let ts = map (C.lookup ctxt) xs in
  let cs = map typeAsProp (combine xs ts) in
    CPropOp (Conj,c::cs)

(* pre: e and e' are numeric expressions *) 
and multiply e e' = match e' with 
  | EVar _ -> e'
  | EConst (Int _) 
  | EConst (Real _) -> E.( * ) e e'
  | EUnaryOp (Neg,e'') -> E.neg (multiply e e'')
  | EBinaryOp (Plus,e1,e2) -> E.(+) (multiply e e1) (multiply e e2)
  | EBinaryOp (Minus,e1,e2) -> E.(-) (multiply e e1) (multiply e e2)
  | EBinaryOp (Mult,e1,e2) -> assert false (* XXX *)

and compileDisj c = c

let compile p = 
(*   assert (isMP p && disjVarsBounded p) *)
  match p with 
    | PMain (d,xts,e,c) -> 
        let xts' = map (fun (x,t) -> (x,compileType t)) xts in 
        let c' = compileProp (C.fromList xts) c in
          PMain (d,xts',e,c')
