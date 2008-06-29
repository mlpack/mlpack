open Ast
open Printf

module E = Edsl
module Id = Util.Id

let showt t = match t with
  | TReal (Discrete (lo,hi)) ->
      let lower = match lo with None -> "-inf" | Some x -> sprintf "%d" x in
      let upper = match hi with None -> "inf" | Some x -> sprintf "%d" x in
        sprintf "[%s,%s]" lower upper
  | TReal (Continuous (lo,hi)) ->
      let lower = match lo with None -> "-inf" | Some x -> sprintf "%f" x in
      let upper = match hi with None -> "inf" | Some x -> sprintf "%f" x in
        sprintf "<%s,%s>" lower upper
  | TBool None -> "bool"
  | TBool (Some b) -> sprintf "{%b}" b

let rec showe e = match e with
  | EVar x -> Id.toString x
  | EConst (Bool b) -> sprintf "%b" b
  | EConst (Real r) -> sprintf "%f" r
  | EConst (Int n) -> sprintf "%d" n
  | EUnaryOp (op,e') -> 
      sprintf "(%s%s)" 
        (match op with Neg -> "-" | Not -> "not ") 
        (showe e')
  | EBinaryOp (op,e1,e2) ->
      sprintf "(%s %s %s)" 
        (showe e1) 
        (match op with Plus -> "+" | Minus -> "-" | Mult -> "*" | Or -> "||" | And -> "&&")
        (showe e2)

let rec showc c = match c with 
  | CBoolVal b -> if b then "T" else "F"
  | CIsTrue e -> sprintf "(isTrue %s)" (showe e)
  | CNumRel (op,e1,e2) ->
      sprintf "(%s %s %s)" 
        (showe e1) 
        (match op with Lte -> "<=" | Equal -> "==" | Gte -> ">=")
        (showe e2)
  | _ -> assert false

;;
printf "%s\n" ( showt E.int' ) ;;
printf "%s\n" ( showt E.real ) ;;
printf "%s\n" ( showt (TBool (Some true)) ) ;;
printf "%s\n" ( showe (E.(&&) (E.(||) E.boolT E.boolF) (E.name "x")) ) ;;
