exception MLFailure of string


(*
  This is the types of binary operation
  For example,it contains +,-,*,/,=,!=,<,<=,&&,||,::
 *)
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


(*
  These are the types of expresion
  For example,it contains Const(int),boolean,var,bin,if,let,letrec,fun,app
 *)
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

(*

 *)
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

(*********************** Some helpers you might need ***********************)

let rec fold f base args = 
  match args with [] -> base
    | h::t -> fold f (f(base,h)) t

let listAssoc (k,l) = 
  fold (fun (r,(t,v)) -> if r = None && k=t then Some v else r) None l

(*********************** Your code starts here ****************************)
(* lookup
 * evironment is used to simulate the evironment in ocaml
 * It is denoted by a list of string * value
 * lookup function is used for find the recent bingding for a variable 
 * because the most common case is that the variable has been assigned for several times.
 *)
let lookup (x,evn) = 
	match listAssoc (x,evn) with
  | Some x -> x
  | None -> raise (MLFailure("variable not bound: " ^ x))

let rec eval (evn,e) = 
  match e with 
  |Bin (e1,op,e2) -> let x = (eval (evn,e1),op,eval (evn,e2)) in
    (match x with
      | (Int e1,Plus,Int e2) -> Int (e1+e2)
      | (Int e1, Minus, Int e2)  -> Int (e1-e2)
      | (Int e1,Mul,Int e2) -> Int (e1*e2)
      | (Int e1,Div,Int e2) -> Int (e1/e2) 
      | (Int e1,Eq,Int e2) -> Bool (e1=e2)
      | (Int e1,Ne,Int e2) -> Bool (e1!=e2)
      | (Int e1,Lt,Int e2) -> Bool (e1<e2)
      | (Int e1,Le,Int e2) -> Bool (e1<=e2)
      | (Bool e1,And,Bool e2) -> Bool (e1&&e2)
      | (Bool e1,Or, Bool e2) -> Bool (e1||e2)
      | (e1, Cons, Nil) -> Pair (e1, Nil)
      | (e1, Cons, e2) -> Pair (e1,e2)
      | _ -> raise (MLFailure ("Binary expression is invalid."))
    )  
  | NilExpr -> Nil
  | Const i -> Int i
  | True -> Bool true
  | False -> Bool false
  | Var x -> (match x with 
              |"hd"   -> NativeFunc "hd"
              |"tl"   -> NativeFunc "tl"
              |"null" -> NativeFunc "null"
              |"map"  -> NativeFunc "map"
              |"foldl"  -> NativeFunc "foldl"
              | _     -> lookup (x,evn))
  | If  (e1,e2,e3) -> (let condi = eval (evn, e1) in
      match condi with
      | Bool true -> eval (evn, e2)
      | Bool false -> eval (evn, e3)
      | _ -> raise (MLFailure "Invalid condition of if statement."))
  | Let (x, e1, e2) -> eval ((x, eval(evn, e1))::evn, e2)
  | Letrec(x, e1, e2) -> (
          match e1 with
        | Fun (_s, expr) -> eval ((x, Closure (evn, Some x, _s, expr)) :: evn, e2)
        | _ -> eval ((x, eval(evn, e1))::evn, e2))
  | Fun(para,body) -> Closure(evn, None, para, body)
  | App(func, input) -> let funVal = eval (evn,func) in
  					    let inputVal = eval (evn,input) in
            match funVal with            
            | NativeFunc "hd" ->   (match inputVal with
                           			 | Pair(h,_) -> h
                            	     | _ -> raise (MLFailure "Input is not a list."))
            | NativeFunc "tl" ->   (match inputVal with
                           			 | Pair (_, Nil) -> Nil
                           			 | Pair (_, Pair (elem, tail)) -> Pair (elem, tail)
                         		     | _ -> raise (MLFailure "Input is not a list."))
            | NativeFunc "null" -> (match inputVal with
                            		 | Nil -> Bool true
                            		 | _ -> Bool false)
            | NativeFunc "map"  -> (match inputVal with
            						 | Closure (evn1, name, para, body)  -> Closure(evn1,Some ("map"),para,body)
            						 | _ ->raise (MLFailure "Map invalid"))
            | NativeFunc "foldl"  -> (match inputVal with
            						 | Closure (evn1, name, para, body)  -> Closure(evn1,Some ("foldl"),para,body)
            						 | _ ->raise (MLFailure "Foldl invalid"))
            | Closure (evn2, name, para, body) -> (match name with
                      					| None -> eval ((para, inputVal)::evn2, body)
                      					| Some "map" -> let myList= inputVal
                      								in 
                      							let rec helperFun l =
                      								match l with
                      								| Pair (h,Pair (elem, tail)) -> Pair(eval ((para,h)::evn2, body),helperFun (Pair (elem, tail)))
                      								| Pair (h,Nil) -> Pair(eval ((para,h)::evn2, body),Nil)
                      								| _ -> raise (MLFailure "map unmatch")
                      							in helperFun myList
                      					| Some "foldl" -> match inputVal with
                      										| Pair(h,_) -> 
                      											let rec helper l = 
                      													match l with
                      													| Pair (h,Pair (elem, tail)) -> let res = helper (Pair (elem, tail)) in let (baseName,_) = List.hd evn2 in eval ((para,h)::(baseName,res)::evn2, body)
                      													| Pair (h,Nil) -> eval ((para,h)::evn2, body)
                      													| _ -> raise (MLFailure "foldl unmatch")
                      										  	in helper inputVal
                      										| _ -> let Closure (evnNext,name,paraNext, bodyNext) = eval ((para,inputVal)::evn2,body) 
                      												in
                      											Closure (evnNext,Some ("foldl"),paraNext, bodyNext) 
				                        | _ -> raise (MLFailure "Closure unmatch")
                      					)
            | _ ->  raise (MLFailure "Apply Unmatch")
  | _ -> raise (MLFailure ("The expression is invalid."))

(**********************     Testing Code  ******************************)

