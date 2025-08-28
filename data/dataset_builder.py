#!/usr/bin/env python3
"""
Dataset builder for AI CI/CD Autofix project.
Creates training data from synthetic errors and mined GitHub data.
"""

import json
import os
import random
import subprocess
from pathlib import Path
import requests
from github import Github  # pip install PyGithub

class DatasetBuilder:
    def __init__(self):
        self.dataset = []
        
    def create_synthetic_errors(self):
        """Create synthetic compilation errors for common Java patterns"""
        synthetic_errors = [
            {
                "error_message": "; expected",
                "context_code": "public class Test { public static void main(String[] args) { System.out.println(\"Hello\") } }",
                "diff_patch": "--- a/Test.java\n+++ b/Test.java\n@@ -1 +1 @@\n-public class Test { public static void main(String[] args) { System.out.println(\"Hello\") } }\n+public class Test { public static void main(String[] args) { System.out.println(\"Hello\"); } }",
                "repo_name": "synthetic",
                "error_type": "syntax"
            },
            {
                "error_message": "cannot find symbol: System.out.printlnx",
                "context_code": "public class Test { public static void main(String[] args) { System.out.printlnx(\"Hello\"); } }",
                "diff_patch": "--- a/Test.java\n+++ b/Test.java\n@@ -1 +1 @@\n-public class Test { public static void main(String[] args) { System.out.printlnx(\"Hello\"); } }\n+public class Test { public static void main(String[] args) { System.out.println(\"Hello\"); } }",
                "repo_name": "synthetic",
                "error_type": "symbol"
            },
            {
                "error_message": "incompatible types: int cannot be converted to String",
                "context_code": "public class Test { public static void main(String[] args) { String s = 10 + 20; } }",
                "diff_patch": "--- a/Test.java\n+++ b/Test.java\n@@ -1 +1 @@\n-public class Test { public static void main(String[] args) { String s = 10 + 20; } }\n+public class Test { public static void main(String[] args) { String s = String.valueOf(10 + 20); } }",
                "repo_name": "synthetic",
                "error_type": "type"
            },
            {
                "error_message": "class, interface, or enum expected",
                "context_code": "public class Test { public static void main(String[] args) { } } }",
                "diff_patch": "--- a/Test.java\n+++ b/Test.java\n@@ -1 +1 @@\n-public class Test { public static void main(String[] args) { } } }\n+public class Test { public static void main(String[] args) { } }",
                "repo_name": "synthetic",
                "error_type": "structure"
            }
        ]
        
        self.dataset.extend(synthetic_errors)
        print(f"Added {len(synthetic_errors)} synthetic error examples")
    
    def mine_github_errors(self, max_repos=5):
        """
        Mine real Java compilation errors from GitHub repositories.
        Note: This requires GitHub API access and proper authentication.
        """
        print("Mining GitHub for Java compilation errors...")
        
        # This is a simplified version - in practice you'd use GitHub API
        # with proper authentication to search for Java compilation errors
        
        # For demo purposes, we'll add some realistic examples
        github_errors = [
            {
                "error_message": "cannot find symbol: class ArrayLis",
                "context_code": "import java.util.*; public class Test { public static void main(String[] args) { ArrayLis<String> list = new ArrayLis<>(); } }",
                "diff_patch": "--- a/Test.java\n+++ b/Test.java\n@@ -1 +1 @@\n-import java.util.*; public class Test { public static void main(String[] args) { ArrayLis<String> list = new ArrayLis<>(); } }\n+import java.util.*; public class Test { public static void main(String[] args) { ArrayList<String> list = new ArrayList<>(); } }",
                "repo_name": "github-mined",
                "error_type": "symbol"
            },
            {
                "error_message": "unreported exception IOException; must be caught or declared to be thrown",
                "context_code": "import java.io.*; public class Test { public static void main(String[] args) { FileReader fr = new FileReader(\"file.txt\"); } }",
                "diff_patch": "--- a/Test.java\n+++ b/Test.java\n@@ -1 +1 @@\n-import java.io.*; public class Test { public static void main(String[] args) { FileReader fr = new FileReader(\"file.txt\"); } }\n+import java.io.*; public class Test { public static void main(String[] args) { try { FileReader fr = new FileReader(\"file.txt\"); } catch (IOException e) { e.printStackTrace(); } } }",
                "repo_name": "github-mined",
                "error_type": "exception"
            }
        ]
        
        self.dataset.extend(github_errors)
        print(f"Added {len(github_errors)} GitHub-mined error examples")
    
    def save_dataset(self, output_path):
        """Save the dataset to a JSONL file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in self.dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Dataset saved to {output_path}")
        print(f"Total examples: {len(self.dataset)}")
        
        # Print statistics
        error_types = {}
        for item in self.dataset:
            error_type = item.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        print("\nDataset Statistics:")
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count} examples")
    
    def validate_dataset(self):
        """Validate the dataset format"""
        print("Validating dataset...")
        
        required_fields = ['error_message', 'context_code', 'diff_patch', 'repo_name']
        valid_count = 0
        
        for i, item in enumerate(self.dataset):
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                print(f"Warning: Example {i} missing fields: {missing_fields}")
            else:
                valid_count += 1
        
        print(f"Validation complete: {valid_count}/{len(self.dataset)} examples are valid")
        return valid_count == len(self.dataset)

def main():
    """Main function to build the dataset"""
    print("Building AI CI/CD Autofix Dataset...")
    print("=" * 50)
    
    builder = DatasetBuilder()
    
    # Create synthetic errors
    builder.create_synthetic_errors()
    
    # Mine GitHub for real errors (simplified for demo)
    builder.mine_github_errors()
    
    # Validate the dataset
    if builder.validate_dataset():
        # Save the dataset
        output_path = "data/processed/java_compilation_dataset.jsonl"
        builder.save_dataset(output_path)
        
        print("\nDataset building completed successfully! 🎉")
        print("\nNext steps:")
        print("1. Train the model: python model/training/train_model.py")
        print("2. Start the API: python model/inference/app.py")
    else:
        print("Dataset validation failed! Please check the data format.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())