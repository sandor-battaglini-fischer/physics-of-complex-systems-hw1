import sympy as sp

def calculate_characteristic_polynomial(matrix):
    """
    Calculate the characteristic polynomial of a square matrix.

    Parameters:
        matrix (sp.Matrix): The input square matrix.

    Returns:
        list: Coefficients of the characteristic polynomial in descending powers of lambda.
    """
    lambda_symbol = sp.symbols('lambda')
    char_poly = (matrix - lambda_symbol * sp.eye(matrix.shape[0])).det()
    poly = sp.Poly(char_poly, lambda_symbol)
    coefficients = poly.all_coeffs()
    return coefficients

def main():
    x, y, z, sigma, rho, b = sp.symbols('x y z sigma rho b')

    matrix_general = sp.Matrix([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -b]
    ])
    matrix_spec = sp.Matrix([
        [-10, 10, 0],
        [rho, -1, 0],
        [0, 0, -8/3]
    ])

    # coeffs = calculate_characteristic_polynomial(matrix_general)
    coeffs = calculate_characteristic_polynomial(matrix_spec)
    print("Characteristic Polynomial Coefficients (from highest degree):")
    for i, coeff in enumerate(coeffs):
        degree = len(coeffs) - i - 1
        if degree > 0:
            print(f"λ^{degree}: {coeff}")
        else:
            print(f"Constant term: {coeff}")

    # Calculate the roots of the characteristic polynomial
    lambda_symbol = sp.symbols('lambda')
    poly_expr = sum(c * lambda_symbol**i for i, c in enumerate(reversed(coeffs)))
    poly = sp.Poly(poly_expr, lambda_symbol)

    try:
        roots = sp.solve(poly_expr, lambda_symbol)
        print("\nRoots of the characteristic polynomial:")
        for i, root in enumerate(roots, start=1):
            print(f"Root {i}: {root}")
    except Exception as e:
        print(f"\nAn error occurred while determining the roots: {e}")

if __name__ == "__main__":
    main()
