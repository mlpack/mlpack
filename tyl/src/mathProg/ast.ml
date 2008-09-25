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

open TylesBase.Util

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
  let eq (x',t') = Id.equal x x' && match t,t' with TReal _ , TReal _ -> true | TBool _ , TBool _ -> true | _ -> false
  in any eq ctxt
      
(* pre: context contains x ; fails otherwise *)
let lookup ctxt x = snd % List.find (Id.equal x % fst) $ ctxt

(* constructors as functions *)
let _Bool x = Bool x
let _Int x = Int x
let _Real x = Real x
let _Discrete x = Discrete x
let _Continuous x = Continuous x
let _TReal x = TReal x
let _TBool x = TBool x
let _EVar x = EVar x
let _EConst x = EConst x
let _EUnaryOp x y = EUnaryOp (x,y)
let _EBinaryOp x y z = EBinaryOp (x,y,z)
let _CBoolVal x = CBoolVal x
let _CIsTrue x = CIsTrue x
let _CNumRel x y z = CNumRel (x,y,z)
let _CPropOp x y = CPropOp (x,y)
let _CQuant w x y z = CQuant (w,x,y,z)
let _PMain w x y z = PMain (w,x,y,z)
