#theory clingo {
		term  {
				- : 5, unary;
				~ : 5, unary;
				** : 4, binary, right;
				* : 3, binary, left;
				/ : 3, binary, left;
				\\ : 3, binary, left;
				+ : 2, binary, left;
				- : 2, binary, left;
				& : 1, binary, left;
				? : 1, binary, left;
				^ : 1, binary, left;
				.. : 0, binary, left
		};
		&eval/0: term, any
}.
