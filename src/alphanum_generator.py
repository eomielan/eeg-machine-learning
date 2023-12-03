"""Script for generating random lowercase alphanumeric strings of specified length."""

import os
import secrets
import string
import time

from colorama import Fore, Style, init

# Constants
STRING_LENGTH = 8
SLEEP_TIME = 2  # seconds


def generate(length: int) -> str:
    """Generate new random lowercase alphanumeric string of given length.

    Args:
        length (int): The length of the string to generate.

    Returns:
        str: The randomly generated string.
    """
    characters = string.ascii_lowercase + string.digits  # Lowercase letters and digits
    random_string = "".join(secrets.choice(characters) for _ in range(length))
    return random_string


def print_diff(expected: str, actual: str) -> None:
    """Print the difference between the expected and actual strings.

    Args:
        expected (str): The expected string.
        actual (str): The actual string.
    """
    correct = True
    coloured_string = ""

    for i in range(min(len(expected), len(actual))):
        if expected[i] == actual[i]:
            coloured_string += Fore.GREEN + actual[i] + Style.RESET_ALL
        else:
            correct = False
            coloured_string += Fore.RED + actual[i] + Style.RESET_ALL

    # Print any remaining characters, if any are present
    if len(actual) > len(expected):
        correct = False
        coloured_string += Fore.RED + actual[len(expected) :] + Style.RESET_ALL
    elif len(expected) > len(actual):
        correct = False
        coloured_string += (
            Fore.RED + "_" * (len(expected) - len(actual)) + Style.RESET_ALL
        )

    if correct:
        print(Fore.GREEN + "CORRECT!" + Style.RESET_ALL)
    else:
        print("Expected: " + Fore.YELLOW + expected + Style.RESET_ALL)
        print("Actual:   ", end="")
        print(coloured_string)


def main():
    init()  # Initialize colorama
    os.system("cls" if os.name == "nt" else "clear")  # Clear screen

    while True:
        input("Press enter to generate a new string")
        os.system("cls" if os.name == "nt" else "clear")

        expected = generate(STRING_LENGTH)
        print(Fore.YELLOW + expected + Style.RESET_ALL)

        time.sleep(SLEEP_TIME)
        os.system("cls" if os.name == "nt" else "clear")

        actual = input("Enter the string: ")
        print("")
        print_diff(expected, actual)

        print("\n")


if __name__ == "__main__":
    main()
