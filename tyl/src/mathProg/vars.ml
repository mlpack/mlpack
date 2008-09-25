open Ast
open Util

module S = Id.Set

let rec freeVarse e = match e with 
  | EVar x              -> S.singleton x
  | EConst _            -> S.empty
  | EUnaryOp (_,e')     -> freeVarse e'
  | EBinaryOp (_,e1,e2) -> S.unions [freeVarse e1; freeVarse e2]

let rec freeVarsc c = match c with 
  | CBoolVal _        -> S.empty
  | CIsTrue e         -> freeVarse e
  | CNumRel (_,e1,e2) -> S.unions [freeVarse e1; freeVarse e2]
  | CPropOp (_,cs)    -> S.unions (map freeVarsc cs)
  | CQuant (_,x,_,c') -> S.remove x (freeVarsc c')

let freeVarsp p = match p with PMain (_,ctxt,e,c) -> 
  let ids = foldl (flip S.add) S.empty % fst % unzip in 
    S.diff $ S.unions [freeVarse e; freeVarsc c] $ ids ctxt

let isClosede = S.is_empty % freeVarse
let isClosedc = S.is_empty % freeVarsc 
let isClosedp = S.is_empty % freeVarsp 

let rec subee e x e' = match e' with 
  | EVar x'              -> if Id.equal x x' then e else e'
  | EConst _             -> e'
  | EUnaryOp (op,e'')    -> EUnaryOp (op, subee e x e'')
  | EBinaryOp (op,e1,e2) -> EBinaryOp (op, subee e x e1, subee e x e2)

let rec subec e x c = match c with 
  | CBoolVal _               -> c
  | CIsTrue e'               -> CIsTrue (subee e x e')
  | CNumRel (op,e1,e2)       -> CNumRel (op, subee e x e1, subee e x e2)
  | CPropOp (op,cs)          -> CPropOp (op, map (subec e x) cs)
  | CQuant (q,x',t,c')       -> (* use alphaConvert here ... *)
      if not (S.mem x (freeVarsc c'))       then c 
      else if not (S.mem x' (freeVarse e))  then CQuant (q,x',t,subec e x c')
      else let x'' = Id.fresh (S.unions [freeVarse e; freeVarsc c']) x' in 
        subec e x (CQuant (q, x'', t, subec (EVar x'') x' c'))

let rec subec' es xs c = match es,xs with
  | e::es',x::xs' -> subec' es' xs' (subec e x c)
  | _ -> c

(* alpha-conversion is a no-op on things that are not variable binders *)
let alphaConvert c taboo = match c with 
  | CQuant(q,x,t,c') -> 
      let x' = Id.fresh (S.union (freeVarsc c) taboo) x 
      in if Id.equal x x' then c else CQuant(q,x',t,subec (EVar x') x c')
  | _ -> c
