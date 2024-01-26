#theory tel {
		formula  {
				<   : 5, unary;         % previous
				<:  : 5, unary;         % weak previous
				<?  : 5, unary;         % eventually-
				<*  : 5, unary;         % always-
				<<  : 5, unary;         % initially
				>   : 5, unary;         % next
				>:  : 5, unary;         % weak next
				>?  : 5, unary;         % eventually+
				>*  : 5, unary;         % always+
				>*  : 4, binary, left;  % release
				>?  : 4, binary, left;  % until
				<*  : 4, binary, left;  % trigger
				<?  : 4, binary, left;  % since
				&   : 3, binary, left;  % and
				|   : 2, binary, left   % or
		};
		&tel/0 : formula, any
}.
