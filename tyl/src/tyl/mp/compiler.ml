open Ast
open List
open Vars
open Wf
open Cnf

module C = Context
module S = Id.Set
module E = Edsl

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
    | CPropOp (Disj,_) -> compileDisj ctxt c
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

and typesAsProp ctxt c = 
  let xs = S.elements (freeVarsc c) in 
  let ts = map (C.lookup ctxt) xs in
  let cs = map typeAsProp (combine xs ts) in
    CPropOp (Conj,c::cs)

(* pre: e and e' are numeric expressions *) 
and scalee e e' = match e' with 
  | EVar _ -> e'
  | EConst (Int _) 
  | EConst (Real _) -> E.( * ) e e'
  | EUnaryOp (Neg,e'') -> E.neg (scalee e e'')
  | EBinaryOp (Plus,e1,e2) -> E.(+) (scalee e e1) (scalee e e2)
  | EBinaryOp (Minus,e1,e2) -> E.(-) (scalee e e1) (scalee e e2)
  | EBinaryOp (Mult, EVar _, _) -> e'
  | EBinaryOp (Mult, _, EVar _) -> e'
  | EBinaryOp (Mult, e1, e2) -> 
      let (x1s,x2s) = (freeVarse e1, freeVarse e2) in
        if S.is_empty x1s && S.is_empty x2s
        then e' 
        else 
          if not (S.is_empty x1s)
          then E.( * ) e1 (scalee e e2)
          else E.( * ) (scalee e e1) e2
  | _ -> failwith "scalee: expression contains non-numeric syntax"

and scalec e c = match c with 
  | CBoolVal _ -> c
  | CIsTrue e' -> E.isTrue (scalee e e') 
  | CNumRel (op,e1,e2) -> CNumRel(op, scalee e e1, scalee e e2)
  | CPropOp (op,cs) -> CPropOp(op, map (scalec e) cs)
  | CQuant (Exists,x,t,c') -> 
      if not (S.mem x (freeVarse e)) 
      then CQuant (Exists,x,t,scalec e c')
      else (* must alpha-convert *)
        let x' = Id.fresh (freeVarse e) x in 
        let c'' = subec (EVar x') x c' in 
          CQuant (Exists,x',t,scalec e c'')

and compileDisj ctxt c = 
  assert (disjVarsBounded ctxt c) ;
  match c with 
    | CPropOp (Disj,cs) -> assert false        
    | _ -> failwith "compileDisj: proposition is not a disjunction"
        
let compile (PMain (d,xts,e,c) as p) = 
  let ctxt = C.fromList xts in
(*     assert (isMP p && disjVarsBounded ctxt c) ;  *)
    let xts' = map (fun (x,t) -> (x,compileType t)) xts in 
    let c' = compileProp ctxt c in
      PMain (d,xts',e,c')
