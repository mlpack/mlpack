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

module Context : sig
  type t 
  val empty : t
  val add : t -> Id.t -> typ -> t
  val fromList : (Id.t * typ) list -> t
  val coarseContains : t -> Id.t -> typ -> bool
  val lookup : t -> Id.t -> typ
end
