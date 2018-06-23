;                     GNU GENERAL PUBLIC LICENSE
;                       Version 3, 29 June 2007
;
; Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
; Everyone is permitted to copy and distribute verbatim copies
; of this license document, but changing it is not allowed.

(def pi 3.141592653589793)

(def e 2.718281828459045)

(def True (= 0 0) )
(def False (= 1 0) )

(defun length (l)
  (__len__ l))

(defun factorial (n)
  (if (= 0 n)
    1
    (* n (factorial (- n 1)))))

(defun gcd (a b)
  (if (= b 0)
    a
    (gcd b (mod a b))))

(defun square (n)
  (* n n))

(defun sqrt (n)
  (exp n 0.5))

(defun head (l)
  (index 0 l))

(defun tail (l)
  (cut 1 l))

(defun abs (x)
  (if (< x 0)
    (- 0 x)
    (x)))

(defun last (l)
  (index (- (length l) 1) l))

(defun last_n (n l)
  (slice (- (length l) n) (length l) l))

(defun slice (a b l)
  (take (- b a) (cut a l)))

(defun not (x)
  (= False x))

(defun neg (n) (- 0 n))

(defun range' (begin end l)
  (if (< begin end)
    (range' (+ 1 begin) end (append begin l))
    l))

(defun range (begin end)
  (range' begin end (list)))

(defun empty (l)
  (= 0 (length l)))

(defun map' (f l a)
  (if (empty l)
    a
    (map' f (tail l) (append (f (head l)) a))))

(defun map (f l)
  (map' f l (list)))

(defun even (n)
  (= 0 (mod n 2)))

(defun filter' (f l a)
  (if (empty l)
    a
    (filter' f (tail l)
      (if (f (head l))
        (append (head l) a)
        a))))

(defun filter (f l)
  (filter' f l (list)))

(defun add (a b) (+ a b))

(defun mul (a b) (* a b))

(defun reduce (f acc l)
  (if (empty l)
    acc
    (reduce f (f acc (head l)) (tail l))))

(defun sum (l)
  (reduce (lambda (a b) (+ a b)) 0 l))

(defun product (l)
  (reduce (lambda (a b) (* a b)) 1 l))

(defun zip' (l1 l2 acc)
  (if (or (empty l1) (empty l2))
    acc
    (zip' (tail l1) (tail l2) (append (list (head l1) (head l2)) acc))))

(defun zip (l1 l2)
  (zip' l1 l2 (list)))

(defun pyramid' (n l)
  (if (= n 0)
    l
    (pyramid' (- n 1) (append (range 0 n) l))))

(defun pyramid (n)
  (pyramid' n (list)))

(defun collatz (x)
  (if (= 1 x)
    (list 1)
    (if (= 0 (mod x 2))
      (append x (collatz (/ x 2)))
      (append x (collatz (+ 1 (* 3 x)))))))

(defun max (l)
  (reduce (lambda (acc x) (if (< acc x) x acc)) (head l) (tail l)))

(defun min (l)
  (reduce (lambda (acc x) (if (> acc x) x acc)) (head l) (tail l)))

(defun quicksort (mylist)
  (if (empty mylist)
    (list)
    (cat (quicksort (filter (lambda (x) (< x (head mylist))) (tail mylist)))
         (list (head mylist))
         (quicksort (filter (lambda (x) (>= x (head mylist)))
                            (tail mylist))))))

(defun fibonacci' (max_n i acc_l)
  (if (= max_n i)
    acc_l
    (fibonacci' max_n (+ i 1) (append (sum (last_n 2 acc_l)) acc_l))))

(defun fibonacci (n)
  (fibonacci' n 0 (list 1)))

(defun isprime' (x acc max_num)
  (if (>= acc max_num)
    True
    (if (= 0 (mod x acc))
      False
      (isprime' x (+ 1 acc) max_num))))

(defun isprime (x)
  (if (<= x 1)
    False
    (isprime' x 2 (+ 1 (sqrt x)))))
