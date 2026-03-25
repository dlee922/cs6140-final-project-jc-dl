"""
Display utilities for consistent output formatting
"""

def print_header(title, width=70, char="="):
    """
    Print a formatted header with a title.
    
    Args:
        title (str): The header text
        width (int): Total width of the header (default: 70)
        char (str): Character to use for the border (default: "=")
    
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


def print_section(title, width=70):
    """
    Print a section separator (lighter than header).
    
    Args:
        title (str): The section text
        width (int): Total width (default: 70)
    
    Example:
        print_section("Step 1: Loading Data")
        # Output:
        # ----------------------------------------------------------------------
        # Step 1: Loading Data
        # ----------------------------------------------------------------------
    """
    print("-" * width)
    print(title)
    print("-" * width)


def print_step(step_num, total_steps, description):
    """
    Print a progress step indicator.
    
    Args:
        step_num (int): Current step number
        total_steps (int): Total number of steps
        description (str): What this step does
    
    Example:
        print_step(1, 5, "Loading data")
        # Output: [Step 1/5] Loading data...
    """
    print(f"\n[Step {step_num}/{total_steps}] {description}...")


def print_success(message):
    """
    Print a success message with checkmark.
    
    Args:
        message (str): Success message
    
    Example:
        print_success("Data loaded successfully")
        # Output: ✓ Data loaded successfully
    """
    print(f"   ✓ {message}")


def print_error(message):
    """
    Print an error message with X mark.
    
    Args:
        message (str): Error message
    
    Example:
        print_error("File not found")
        # Output: ✗ File not found
    """
    print(f"   ✗ {message}")


def print_info(message, indent=True):
    """
    Print an informational message.
    
    Args:
        message (str): Info message
        indent (bool): Whether to indent (default: True)
    
    Example:
        print_info("Found 566 samples")
        # Output:    Found 566 samples
    """
    prefix = "   " if indent else ""
    print(f"{prefix}{message}")


def print_summary(title, items, width=70):
    """
    Print a formatted summary with multiple items.
    
    Args:
        title (str): Summary title
        items (dict): Dictionary of label: value pairs
        width (int): Total width (default: 70)
    
    Example:
        print_summary("Data Summary", {
            "Mutations": "157,145",
            "Samples": "566",
            "Patients": "566"
        })
        # Output:
        # ======================================================================
        # Data Summary
        # ======================================================================
        # Mutations: 157,145
        # Samples: 566
        # Patients: 566
    """
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    for label, value in items.items():
        print(f"{label}: {value}")


def print_dataframe_info(df, name="DataFrame"):
    """
    Print useful information about a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to describe
        name (str): Name of the DataFrame for display
    
    Example:
        print_dataframe_info(mutations_df, "Mutations")
        # Output:
        # Mutations shape: (157145, 28)
        # Columns: uniqueSampleKey, uniquePatientKey, ...
    """
    print(f"{name} shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns[:5].tolist())}...")
    if len(df.columns) > 5:
        print(f"  (and {len(df.columns) - 5} more)")