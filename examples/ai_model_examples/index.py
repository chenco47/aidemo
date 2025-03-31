"""
Index file to demonstrate all available AI model examples.
"""
import os
import sys
import importlib.util

# Make sure the parent directory is in the path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def get_module_description(module_name):
    """Get the module description from the module docstring."""
    try:
        module_path = os.path.join(current_dir, f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return "Module not found"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        docstring = module.__doc__ or "No description available"
        return docstring.strip()
    except Exception as e:
        return f"Error loading module: {str(e)}"

def list_example_files():
    """List all example files in the directory."""
    example_files = []
    
    for file in os.listdir(current_dir):
        if file.endswith(".py") and file != "__init__.py" and file != "index.py":
            module_name = file[:-3]  # Remove .py extension
            description = get_module_description(module_name)
            example_files.append((module_name, description))
    
    return example_files

def display_model_examples():
    """Display information about all available model examples."""
    examples = list_example_files()
    
    print("=" * 80)
    print("AI MODEL EXAMPLES DIRECTORY".center(80))
    print("=" * 80)
    print(f"\nFound {len(examples)} example files:\n")
    
    for i, (module_name, description) in enumerate(examples, 1):
        print(f"{i}. {module_name}.py")
        print(f"   {description}")
        print()
    
    print("=" * 80)
    print("USAGE INSTRUCTIONS".center(80))
    print("=" * 80)
    print("\nTo run an example file, use:")
    print(f"    python -m aidemo.examples.ai_model_examples.<filename>\n")
    print("Example:")
    print(f"    python -m aidemo.examples.ai_model_examples.{examples[0][0] if examples else 'example_file'}\n")
    print("Note: Most examples require API keys and will not perform actual API calls by default.")
    print("      Uncomment the relevant code in the main section of each file to make real API calls.")
    print("=" * 80)

def show_model_usage(module_name):
    """Show detailed usage for a specific model example."""
    try:
        module_path = os.path.join(current_dir, f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            print(f"Module {module_name} not found")
            return
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print("=" * 80)
        print(f"USAGE FOR: {module_name}.py".center(80))
        print("=" * 80)
        
        # Print docstring
        if module.__doc__:
            print(f"\n{module.__doc__.strip()}\n")
        
        # Print available functions
        functions = [name for name, obj in vars(module).items() 
                    if callable(obj) and not name.startswith('_')]
        
        if functions:
            print("Available functions:")
            for func_name in functions:
                func = getattr(module, func_name)
                if func.__doc__:
                    print(f"  - {func_name}: {func.__doc__.strip()}")
                else:
                    print(f"  - {func_name}")
        
        print("\nTo run this example:")
        print(f"    python -m aidemo.examples.ai_model_examples.{module_name}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error loading module: {str(e)}")

def main():
    """Main function to run the index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Model Examples Index")
    parser.add_argument("--list", action="store_true", help="List all available examples")
    parser.add_argument("--show", type=str, help="Show detailed usage for a specific example")
    
    args = parser.parse_args()
    
    if args.show:
        show_model_usage(args.show)
    else:
        # Default to listing all examples
        display_model_examples()

if __name__ == "__main__":
    main() 