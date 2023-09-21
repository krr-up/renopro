#theory dc {
  constant {
    - : 0, unary
  };
  diff_term {
    - : 0, binary, left
  };
  domain_term {
    .. : 1, binary, left
  };
  show_term {
    / : 1, binary, left
  };
  &dom/0: domain_term, { = }, diff_term, any;
  &diff/0: diff_term, { <= }, constant, any;
  &show/0: show_term, directive
}.
