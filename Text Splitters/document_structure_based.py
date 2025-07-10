from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = '''
# Sample Python Program: Factorial Calculator

def factorial(n):
    """Returns the factorial of a given non-negative integer."""
    if n < 0:
        return "Factorial is not defined for negative numbers."
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Main function
def main():
    try:
        number = int(input("Enter a non-negative integer: "))
        print(f"The factorial of {number} is {factorial(number)}")
    except ValueError:
        print("Invalid input! Please enter a valid integer.")

# Entry point
if __name__ == "__main__":
    main()

'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 350,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])