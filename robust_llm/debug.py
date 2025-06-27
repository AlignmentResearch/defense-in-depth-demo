import subprocess

# Define the command as a list of strings
command = ["python", "-m", "robust_llm", "+experiment=niki/2_DEBUG_niki_130"]

# Run the command
result = subprocess.run(command, capture_output=True, text=True)
