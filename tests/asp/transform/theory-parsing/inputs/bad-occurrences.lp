#theory bad {
		formula_head  {
				+   : 1, unary
		};
		formula_body  {
				-   : 1, unary
		};
		&a/0 : formula_head, head;
		&b/0 : formula_body, body;
		&c/0 : formula_head, directive
}.

&a { + 1 }.
&b { - 2 } :- d.
&c { + 3 } :- e.

