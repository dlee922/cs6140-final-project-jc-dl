"""
Display utilities for consistent output formatting
"""

def print_header(title, width=70, char="="):
    """
    Print a formatted header with a title.

    Example:
        print_header("Data Processing")
        # Output:
        # ======================================================================
        # Data Processing
        # ======================================================================
    """
    print(char * width)
    print(title)
    print(char * width)


def print_step(step_num, total_steps, description):
    """
    Print a progress step indicator.

    Example:
        print_step(1, 5, "Loading data")
        # Output: [Step 1/5] Loading data...
    """
    print(f"\n[Step {step_num}/{total_steps}] {description}...")


def print_success(message):
    """
    Print a success message
    
    Example:
        print_success("Data loaded successfully")
        # Output: Data loaded successfully
    """
    print(f"{message}")


def print_info(message, indent=True):
    """
    Print an informational message.

    Example:
        print_info("Found 566 samples")
        # Output:    Found 566 samples
    """
    prefix = "   " if indent else ""
    print(f"{prefix}{message}")
