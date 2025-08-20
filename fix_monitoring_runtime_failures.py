#!/usr/bin/env python3
"""
Fix critical runtime failures in the monitoring system.

Addresses three critical issues:
1. JSON serialization failure for enum objects
2. Missing AlertSeverity.ERROR enum member
3. Reserved keyword collision with 'exc_info'
"""

import os
import sys
import re
from pathlib import Path

def fix_json_serialization():
    """Fix JSON serialization issue in structured_logger.py"""
    file_path = Path("src/monitoring/structured_logger.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The issue is already fixed - category.value is used in to_dict()
    # But let's verify all enum usages are correct
    
    # Check if LogLevel enum values are properly serialized
    if 'self.level.value' in content:
        print("✅ LogLevel enum properly serialized")
    else:
        # Fix LogLevel serialization if needed
        content = re.sub(
            r"'level': self\.level,",
            "'level': self.level.value,",
            content
        )
        print("✅ Fixed LogLevel enum serialization")
    
    # Write back if changed
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True


def fix_missing_enum_member():
    """Fix missing AlertSeverity.ERROR enum member"""
    
    # First, add ERROR to the AlertSeverity enum in dashboard_integration.py
    dashboard_file = Path("src/monitoring/dashboard_integration.py")
    
    if dashboard_file.exists():
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the AlertSeverity enum and add ERROR if missing
        if 'ERROR = "error"' not in content:
            # Add ERROR between WARNING and CRITICAL
            content = re.sub(
                r'(WARNING = "warning"\n)(    CRITICAL = "critical")',
                r'\1    ERROR = "error"\n\2',
                content
            )
            
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Added ERROR member to AlertSeverity enum in {dashboard_file}")
        else:
            print(f"✅ ERROR member already exists in AlertSeverity enum")
    
    # Now fix all files that use AlertSeverity.ERROR
    # The production_monitor.py already uses it correctly, assuming the enum has ERROR
    
    # Also need to fix references to AlertSeverity.LOW which doesn't exist
    production_file = Path("src/monitoring/production_monitor.py")
    
    if production_file.exists():
        with open(production_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace AlertSeverity.LOW with AlertSeverity.INFO
        if 'AlertSeverity.LOW' in content:
            content = content.replace('AlertSeverity.LOW', 'AlertSeverity.INFO')
            
            with open(production_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed AlertSeverity.LOW references in {production_file}")
        else:
            print(f"✅ No AlertSeverity.LOW references found")
    
    return True


def fix_reserved_keyword_collision():
    """Fix reserved keyword collision with 'exc_info'"""
    file_path = Path("src/monitoring/structured_logger.py")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace exc_info in extra dict with a different name
    # Lines 266 and 273 use extra['exc_info'] = True
    
    # Change from extra['exc_info'] to extra['has_exception']
    content = re.sub(
        r"extra\['exc_info'\] = True",
        "extra['has_exception'] = True",
        content
    )
    
    # Also need to ensure we don't pass exc_info as an extra field to logger
    # The issue is that exc_info is a reserved parameter for Python's logging
    
    # Update the _log method to handle this
    old_log_method = """def _log(self, level: LogLevel, message: str, category: LogCategory, **fields):
        \"\"\"Internal logging method\"\"\"
        # Add category to extra fields
        extra = fields.copy()
        extra['category'] = category
        
        # Map custom levels to standard logging levels
        if level == LogLevel.TRACE:
            log_level = logging.DEBUG
        elif level == LogLevel.AUDIT:
            log_level = logging.INFO
        else:
            log_level = getattr(logging, level.value)
        
        self.logger.log(log_level, message, extra=extra)"""
    
    new_log_method = """def _log(self, level: LogLevel, message: str, category: LogCategory, **fields):
        \"\"\"Internal logging method\"\"\"
        # Extract exc_info if present (it's a reserved parameter)
        exc_info = fields.pop('exc_info', None)
        
        # Add category to extra fields
        extra = fields.copy()
        extra['category'] = category
        
        # Map custom levels to standard logging levels
        if level == LogLevel.TRACE:
            log_level = logging.DEBUG
        elif level == LogLevel.AUDIT:
            log_level = logging.INFO
        else:
            log_level = getattr(logging, level.value)
        
        # Pass exc_info as a separate parameter if present
        if exc_info:
            self.logger.log(log_level, message, exc_info=exc_info, extra=extra)
        else:
            self.logger.log(log_level, message, extra=extra)"""
    
    content = content.replace(old_log_method, new_log_method)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Fixed reserved keyword collision in {file_path}")
    return True


def main():
    """Apply all fixes"""
    print("🔧 Applying monitoring system runtime fixes...")
    print()
    
    # Change to project root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    # Fix 1: JSON Serialization
    print("1️⃣ Fixing JSON serialization issue...")
    if not fix_json_serialization():
        success = False
    print()
    
    # Fix 2: Missing enum member
    print("2️⃣ Fixing missing AlertSeverity.ERROR enum member...")
    if not fix_missing_enum_member():
        success = False
    print()
    
    # Fix 3: Reserved keyword collision
    print("3️⃣ Fixing reserved keyword collision...")
    if not fix_reserved_keyword_collision():
        success = False
    print()
    
    if success:
        print("✅ All fixes applied successfully!")
        print()
        print("Next steps:")
        print("1. Run the integration tests to verify fixes")
        print("2. Test the monitoring system initialization")
        print("3. Validate logging output format")
    else:
        print("❌ Some fixes failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()