open Ast
open Util

module S = Id.Set

let rec freeVarse e = match e with 
  | EVar x              -> S.singleton x
  | EConst _            -> S.empty
  | EUnaryOp (_,e')     -> freeVarse e'
  | EBinaryOp (_,e1,e2) -> S.union' [freeVarse e1; freeVarse e2]

let rec freeVarsc c = match c with 
  | CBoolVal _        -> S.empty
  | CIsTrue e         -> freeVarse e
  | CNumRel (_,e1,e2) -> S.union' [freeVarse e1; freeVarse e2]
  | CPropOp (_,cs)    -> S.union' (map freeVarsc cs)
  | CQuant (_,x,_,c') -> S.remove x (freeVarsc c')

let freeVarsp p = match p with PMain (_,ctxt,e,c) -> 
  let ids = foldl (flip S.add) S.empty % fst % unzip in 
    S.diff $ S.union' [freeVarse e; freeVarsc c] $ ids ctxt

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
      else let x'' = Id.fresh (S.union' [freeVarse e; freeVarsc c']) x' in 
        subec e x (CQuant (q, x'', t, subec (EVar x'') x' c'))

let rec subec' es xs c = match es,xs with
  | e::es',x::xs' -> subec' es' xs' (subec e x c)
  | _ -> c

(* the rename/alphaConvert functions replace all *free* occurences of
   x with x' ... pre: x' is not free in e *)

let rec renamee x x' e = match e with 
  | EVar x''             -> if Id.equal x x'' then EVar x' else e
  | EConst _             -> e
  | EUnaryOp (op,e')     -> EUnaryOp (op, renamee x x' e')
  | EBinaryOp (op,e1,e2) -> EBinaryOp (op, renamee x x' e1, renamee x x' e2)
  
let rec renamec x x' c = match c with
  | CBoolVal _               -> c
  | CIsTrue e                -> CIsTrue (renamee x x' e)
  | CNumRel (op,e1,e2)       -> CNumRel (op,renamee x x' e1,renamee x x' e2)
  | CPropOp (op,cs)          -> CPropOp (op,map (renamec x x') cs)
  | CQuant (Exists,x'',t,c') -> if Id.equal x'' x then c else CQuant (Exists,x'',t,renamec x x' c')

let alphaConverte x e taboo = let x' = Id.fresh (S.union (freeVarse e) taboo) x in x', renamee x x' e
let alphaConvertc x c taboo = let x' = Id.fresh (S.union (freeVarsc c) taboo) x in x', renamec x x' c
