# Mufi
Implementation for the Mufi Programming Language

Mufi is a functional, lisp-like language

An example of Mufi syntax can be seen below with the standard library implementation of quicksort
```
(defun quicksort (mylist)                                                       
  (if (empty mylist)                                                            
    (list)                                                                      
    (cat (quicksort (filter (lambda (x) (< x (head mylist))) (tail mylist)))
         (list (head mylist))
         (quicksort (filter (lambda (x) (>= x (head mylist)))
                            (tail mylist))))))
```
Or below is how one could implement the collatz conjecture
```
(defun collatz (x)
  (if (= 1 x)
    (list 1)
    (if (= 0 (mod x 2))
      (append x (collatz (/ x 2)))
      (append x (collatz (+ 1 (* 3 x)))))))
```

Mufi is interactive, and includes a REPL for experimentation

```
> (filter isprime (range 0 15))
[2: NUMBER, 3: NUMBER, 5: NUMBER, 7: NUMBER, 11: NUMBER, 13: NUMBER]: LIST
```

With Mufi being a functional language, it is well suited for solving mathematical problems, such as those found on https://projecteuler.net/
```
> (sum (filter even (filter (lambda (x) (< x 4000000)) (fibonacci 1000))))
4613732: NUMBER
```

Mufi is currently implemented in Python3 and works well with pypy3. There are current plans to implement Mufi in a higher performance language such as Rust in the future.

Tests are run as follows:
```bash
$ ./mufi.py --test
  .
  .
  .
  57 PASS Input: '(sum (filter even (filter (lambda (x) (< x 4000000)) (fibonacci 1000))))' -> '4613732: NUMBER'
  58 PASS Input: '(filter isprime (range (neg 5) 20))' -> '[2: NUMBER, 3: NUMBER, 5: NUMBER, 7: NUMBER, 11: NUMBER, 13: NUMBER, 17: NUMBER, 19: NUMBER]: LIST'


ALL PASSING, 58/58 Passed

```
