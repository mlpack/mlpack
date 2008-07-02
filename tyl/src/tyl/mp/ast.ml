(* Naming conventions:
   i - index
   e - expr
   t - type
   p - prog
   c - prop
   z - prop type
   x - variable/id
   cs,xs - plural (sets/lists of things)
*)

open List

type nullOp = Bool of bool | Int of int | Real of float
type unaryOp = Neg | Not 
type binaryOp = Plus | Minus | Mult | Or | And
type numRel = Equal | Lte | Gte
type propOp = Disj | Conj
type quant = Exists
type direction = Min | Max


type 'a interval = 'a option * 'a option
type refinedReal = Discrete of int interval | Continuous of float interval 
type refinedBool = bool option

(* this representation for 'typ' was chosen so that the data
   constructors are also the coarse types of the object language *)
type typ = 
  | TReal of refinedReal
  | TBool of refinedBool
      
type prop_typ = 
  | ZProp

type expr = 
  | EVar of Id.t
  | EConst of nullOp
  | EUnaryOp of unaryOp * expr
  | EBinaryOp of binaryOp * expr * expr

type prop = 
  | CBoolVal of bool
  | CIsTrue of expr
  | CNumRel of numRel * expr * expr
  | CPropOp of propOp * prop list
  | CQuant of quant * Id.t * typ * prop
      
type prog = 
  | PMain of direction * ((Id.t * typ) list) * expr * prop

type context = (Id.t * typ) list

let coarseContains ctxt x t = 
  let coarseEqual (x',t') = Id.equal x x' &&
    match t,t' with
      | TReal _ , TReal _ -> true
      | TBool _ , TBool _ -> true
      | _ -> false
  in 
    exists coarseEqual ctxt
      
(* pre: context contains x ; fails otherwise *)
let lookup ctxt x = snd (find (fun (x',_) -> Id.equal x x') ctxt)
