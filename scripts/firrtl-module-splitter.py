import os
import argparse

def split_firrtl_modules(input_file, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the input file
    with open(input_file, "r") as file:
        lines = file.readlines()

    current_module = None
    current_lines = []
    modules = []

    # Parse the file
    for line in lines:
        stripped_line = line.strip()

        # Check for module or extmodule declaration
        if stripped_line.startswith("module ") or stripped_line.startswith("extmodule "):
            # Save the previous module if it exists
            if current_module:
                modules.append((current_module, current_lines))

            # Start a new module
            parts = stripped_line.split()
            current_module = parts[1].strip(":")
            current_lines = [line[2:]]
        elif current_module:
            current_lines.append(line[2:])

    # Save the last module
    if current_module:
        modules.append((current_module, current_lines))

    # Write each module to a separate file
    for module_name, module_lines in modules:
        module_file_path = os.path.join(output_dir, f"{module_name}.fir")
        with open(module_file_path, "w") as module_file:
            module_file.writelines(module_lines)
        print(f"Module {module_name} saved to {module_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split FIRRTL modules into separate files.")
    parser.add_argument("input_file", help="Path to the input FIRRTL file.")
    parser.add_argument("output_dir", help="Directory to save the split module files.")
    args = parser.parse_args()

    split_firrtl_modules(args.input_file, args.output_dir)

