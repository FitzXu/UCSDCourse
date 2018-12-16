(* CSE 130: Programming Assignment 1
 * misc.ml
 *)

(* sumList : int list -> int 
   Function: calcualate the sum of a int list
   Logic Behind: 
   Sum(List[0..n]) = List[0] + Sum(List[1..n])
   Therefore, use head:tailList to seperate the list
*) 

let rec sumList l = 
	match l with
	| []->0
	| head::tailList -> head + sumList tailList;;


(* digitsOfInt : int -> int list 
   Function: convert an integer into a list composed by its digits
   Logig Behind:
   Assume an Integer a1a2...an
   digitsOfInt(a1a2...an) =  digitsOfInt(a1a2...an-1) + [an]
   an can be obtained by mod 10, and a1a2...an-1 can be obtained by /10.
   However, we can't use @ operator, so I implemented a function named app.
   (see the digits function below for an example of what is expected)
 *)

let rec digitsOfInt n = 
  let rec app l i = 
  match l with 
  | [] -> [i]
  | h::t -> h::(app t i) in
	if n <10 then [n]
	else app (digitsOfInt (n /10)) (n mod 10);;


(* digits : int -> int list
 * (digits n) is the list of digits of n in the order in which they appear
 * in n
 * e.g. (digits 31243) is [3,1,2,4,3]
 *      (digits (-23422) is [2,3,4,2,2]

 *)
 
let digits n = digitsOfInt (abs n);;


(* From http://mathworld.wolfram.com/AdditivePersistence.html
 * Consider the process of taking a number, adding its digits, 
 * then adding the digits of the number derived from it, etc., 
 * until the remaining number has only one digit. 
 * The number of additions required to obtain a single digit from a number n 
 * is called the additive persistence of n, and the digit obtained is called 
 * the digital root of n.
 * For example, the sequence obtained from the starting number 9876 is (9876, 30, 3), so 
 * 9876 has an additive persistence of 2 and a digital root of 3.
 *)


(* ***** PROVIDE COMMENT BLOCKS FOR THE FOLLOWING FUNCTIONS ***** *)

(* additivePersistence : int -> int 
   Function: calcualate the additive persistence
   Logic Behind: 
   set a counter to accumulate the times to form a one-digit number
   additivePersistence(n) = 1 + additivePersistence(the number summated by List(n))
   we can call sumList to summate and call digitsOfInt to form a list
*)
let rec additivePersistence n = 
  if n<10 then 0
  else 1 + additivePersistence (sumList (digitsOfInt n));;

(* digitalRoot : int -> int
   Function: the final one-digit number of the above process
   Logic Behind(recursive formula):
   digitalRoot(n) = digitalRoot(the number summated by List(n))
   termination condition: when n < 10, it means it is already a one-digit number
*)
let rec digitalRoot n = 
  if n<10 then n
  else digitalRoot (sumList (digitsOfInt n));;

(* listReverse : 'a list -> 'a list
   Function: reverse a list
   Login Behind(recursive formula):
   L[n...0] = L[n...1] + L[0]
   As Stated above, we implemented auxiliary function app.
*)
let rec listReverse l = 
  let rec app l i = 
  match l with 
  | [] -> [i]
  | h::t -> h::(app t i) in
  match l with 
  | [] -> []
  | h::t -> app (listReverse t) h;;
   

(* explode : string -> char list 
 * (explode s) is the list of characters in the string s in the order in 
 *   which they appear
 * e.g.  (explode "Hello") is ['H';'e';'l';'l';'o']
 *)
let explode s = 
  let rec _exp i = 
    if i >= String.length s then [] else (s.[i])::(_exp (i+1)) in
  _exp 0

(* reverse : string -> string
   Function: reverse a string
   e.g. "abcd" -> "dcba"
   Logic behind(Recursive Function):
   string[n...0] = string[n...1] ^ string[0]
   Since s.[i] return type char, 
   so we use String.make casting into type String to use operation ^
*)
let reverse s =
  let rec helper i =
    if i >= String.length s then "" else (helper (i+1))^(String.make 1 s.[i]) 
    in 
    helper 0

(* palindrom: string -> bool
   Function: judge whether the input string is palindrome
   Logic behind:
   A palindrome string should has the following structure:
   s[0]s[1]...s[n] = s[n]s[n-1]...s[0]
   So we can judge where the explosion list of input string and reversed string are the same
*)
let palindrome w = 
  if (explode w) = (explode (reverse w)) then true
  else false ;;

(************** Add Testing Code Here ***************)
