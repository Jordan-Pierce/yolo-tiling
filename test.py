import argparse

def add_numbers(a, b):
    """
    Add two numbers together and return their sum.

    Args:
        a (float): First number to add
        b (float): Second number to add

    Returns:
        float: The sum of a and b
    """
    return a + b

def subtract_numbers(a, b):
    """
    Subtract b from a and return the difference.

    Args:
        a (float): Number to subtract from
        b (float): Number to subtract

    Returns:
        float: The difference of a and b
    """
    return a - b

def multiply_numbers(a, b):
    """
    Multiply two numbers together and return their product.

    Args:
        a (float): First number to multiply
        b (float): Second number to multiply

    Returns:
        float: The product of a and b
    """
    return a * b

def divide_numbers(a, b):
    """
    Divide a by b and return the quotient.

    Args:
        a (float): Numerator
        b (float): Denominator

    Returns:
        float: The quotient of a and b
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def get_valid_number(prompt):
    """Get a valid number input from the user."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("That's not a valid number. Please try again.")

def get_operation():
    """Get a valid operation choice from the user."""
    valid_operations = {
        '1': 'add',
        '2': 'subtract',
        '3': 'multiply',
        '4': 'divide'
    }
    
    while True:
        print("\nAvailable operations:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        choice = input("\nChoose an operation (1-4): ")
        
        if choice in valid_operations:
            return valid_operations[choice]
        else:
            print("Invalid choice. Please select 1-4.")

def main():
    while True:
        print("\n== Calculator ==")
        
        # Get first number
        num1 = get_valid_number("Enter first number: ")
        
        # Get second number
        num2 = get_valid_number("Enter second number: ")
        
        # Get operation
        operation = get_operation()
        
        # Perform selected operation
        if operation == 'add':
            result = add_numbers(num1, num2)
            operation_symbol = "+"
        elif operation == 'subtract':
            result = subtract_numbers(num1, num2)
            operation_symbol = "-"
        elif operation == 'multiply':
            result = multiply_numbers(num1, num2)
            operation_symbol = "*"
        elif operation == 'divide':
            try:
                result = divide_numbers(num1, num2)
                operation_symbol = "/"
            except ValueError as e:
                print(f"Error: {e}")
                continue
        
        # Print result
        print(f"\n{num1} {operation_symbol} {num2} = {result}")
        
        # Ask if they want to continue
        play_again = input("\nWould you like to perform another calculation? (yes/no): ").lower()
        if play_again not in ['yes', 'y']:
            print("Thank you for using the calculator!")
            break

if __name__ == "__main__":
    main()
