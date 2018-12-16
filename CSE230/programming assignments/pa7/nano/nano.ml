exception MLFailure of string

type binop = 
  Plus 
| Minus 
| Mul 
| Div 
| Eq 
| Ne 
| Lt 
| Le 
| And 
| Or          
| Cons

type expr =   
  Const of int 
| True   
| False      
| NilExpr
| Var of string    
| Bin of expr * binop * expr 
| If  of expr * expr * expr
| Let of string * expr * expr 
| App of expr * expr 
| Fun of string * expr    
| Letrec of string * expr * expr
	
type value =  
  Int of int		
| Bool of bool          
| Closure of env * string option * string * expr 
| Nil                    
| Pair of value * value     
| NativeFunc of string

and env = (string * value) list

let binopToString op = 
  match op with
      Plus -> "+" 
    | Minus -> "-" 
    | Mul -> "*" 
    | Div -> "/"
    | Eq -> "="
    | Ne -> "!="
    | Lt -> "<"
    | Le -> "<="
    | And -> "&&"
    | Or -> "||"
    | Cons -> "::"

let rec valueToString v = 
  match v with 
    Int i -> 
      Printf.sprintf "%d" i
  | Bool b -> 
      Printf.sprintf "%b" b
  | Closure (evn,fo,x,e) -> 
      let fs = match fo with None -> "Anon" | Some fs -> fs in
      Printf.sprintf "{%s,%s,%s,%s}" (envToString evn) fs x (exprToString e)
  | Pair (v1,v2) -> 
      Printf.sprintf "(%s::%s)" (valueToString v1) (valueToString v2) 
  | Nil -> 
      "[]"
  | NativeFunc s ->
      Printf.sprintf "Native %s" s

and envToString evn =
  let xs = List.map (fun (x,v) -> Printf.sprintf "%s:%s" x (valueToString v)) evn in
  "["^(String.concat ";" xs)^"]"

and exprToString e =
  match e with
      Const i ->
        Printf.sprintf "%d" i
    | True -> 
        "true" 
    | False -> 
        "false"
    | NilExpr -> 
        "[]"
    | Var x -> 
        x
    | Bin (e1,op,e2) -> 
        Printf.sprintf "%s %s %s" 
        (exprToString e1) (binopToString op) (exprToString e2)
    | If (e1,e2,e3) -> 
        Printf.sprintf "if %s then %s else %s" 
        (exprToString e1) (exprToString e2) (exprToString e3)
    | Let (x,e1,e2) -> 
        Printf.sprintf "let %s = %s in \n %s" 
        x (exprToString e1) (exprToString e2) 
    | App (e1,e2) -> 
        Printf.sprintf "(%s %s)" (exprToString e1) (exprToString e2)
    | Fun (x,e) -> 
        Printf.sprintf "fun %s -> %s" x (exprToString e) 
    | Letrec (x,e1,e2) -> 
        Printf.sprintf "let rec %s = %s in \n %s" 
        x (exprToString e1) (exprToString e2) 

and exprToPython e =
  match e with
      Const i ->
        Printf.sprintf "Const(%d)" i
    | True -> 
        "Bool(True)" 
    | False -> 
        "Bool(False)"
    | NilExpr -> 
        "NilExpr()"
    | Var x -> 
        Printf.sprintf "Var(\"%s\")" x
    | Bin (e1,op,e2) -> 
        Printf.sprintf "Bin(%s,%s,%s)" 
        (exprToPython e1) (binopToPython op) (exprToPython e2)
    | If (e1,e2,e3) -> 
        Printf.sprintf "If(%s,%s,%s)" 
        (exprToPython e1) (exprToPython e2) (exprToPython e3)
    | Let (x,e1,e2) -> 
        Printf.sprintf "Let(\"%s\",%s,%s)" 
        x (exprToPython e1) (exprToPython e2) 
    | App (e1,e2) -> 
        Printf.sprintf "App(%s,%s)" (exprToPython e1) (exprToPython e2)
    | Fun (x,e) -> 
        Printf.sprintf "Fun(\"%s\",%s)" x (exprToPython e) 
    | Letrec (x,e1,e2) -> 
        Printf.sprintf "Letrec(\"%s\",%s,%s)" 
        x (exprToPython e1) (exprToPython e2) 
and binopToPython op = 
  match op with
      Plus -> "Plus()" 
    | Minus -> "Minus()" 
    | Mul -> "Mul()" 
    | Div -> "Div()"
    | Eq -> "Eq()"
    | Ne -> "Neq()"
    | Lt -> "Lt()"
    | Le -> "Leq()"
    | And -> "And()"
    | Or -> "Or()"
    | Cons -> "Cons()"
