def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places

    p(x|y)=p(x ^ y) / p(y)


    """
    count_xy = 0
    count_x = 0

    for X, Y in data:
        if X == x:
            count_x += 1

            if Y == y:
                count_xy += 1

    if count_x == 0:
        return 0.0

    return round(count_xy / count_x, 4)