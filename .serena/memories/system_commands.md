# System Commands for Linux Environment

## File Operations
```bash
# List files and directories
ls -la

# Navigate directories  
cd /path/to/directory

# File operations
cp source_file destination
mv old_name new_name
rm file_name
mkdir directory_name

# File content viewing
cat filename
head -n 20 filename
tail -n 20 filename
less filename
```

## Search Operations  
```bash
# Search for text in files
grep -r "search_term" directory/

# Find files by name
find . -name "*.py" -type f

# Search with ripgrep (if available)
rg "search_pattern" --type py
```

## Git Operations
```bash
# Status and basic operations
git status
git add .
git commit -m "message"
git push origin main

# Branch operations
git branch
git checkout -b new_branch
git merge branch_name

# View logs
git log --oneline -10
```

## Python Environment
```bash
# Python version check
python3 --version

# Install dependencies
pip install -r requirements.txt

# Virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv/Scripts/activate     # Windows

# Module execution
python3 -m pytest
python3 -c "import statement; code"
```

## Process Management
```bash
# View running processes
ps aux | grep python

# Kill process by PID
kill PID

# Background execution
command &

# Check disk space
df -h

# Check memory usage
free -h
```

## File Permissions
```bash
# Make executable
chmod +x script.py

# Change permissions
chmod 755 directory/
chmod 644 file.txt
```

## Environment Variables
```bash
# Set temporarily
export PYTHONPATH="$PWD/src"

# View current environment
env | grep PYTHON
```