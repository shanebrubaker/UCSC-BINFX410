#!/usr/bin/env python3
"""
Example 02: Simple Calculator Module

A basic calculator module demonstrating function design,
type hints, and error handling.
"""


def add(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return the quotient of two numbers.

    Args:
        a: The dividend
        b: The divisor

    Returns:
        The quotient of a divided by b

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base: float, exponent: float) -> float:
    """Return base raised to the power of exponent."""
    return base ** exponent


def modulo(a: float, b: float) -> float:
    """Return the remainder of a divided by b.

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot compute modulo with zero divisor")
    return a % b


# Interactive CLI example
if __name__ == "__main__":
    print("Simple Calculator")
    print("Operations: add, subtract, multiply, divide, power, modulo")
    print("Type 'quit' to exit\n")

    operations = {
        'add': add,
        'subtract': subtract,
        'multiply': multiply,
        'divide': divide,
        'power': power,
        'modulo': modulo,
    }

    while True:
        op = input("Enter operation: ").strip().lower()

        if op == 'quit':
            print("Goodbye!")
            break

        if op not in operations:
            print(f"Unknown operation: {op}")
            continue

        try:
            a = float(input("Enter first number: "))
            b = float(input("Enter second number: "))
            result = operations[op](a, b)
            print(f"Result: {result}\n")
        except ValueError as e:
            print(f"Error: {e}\n")
