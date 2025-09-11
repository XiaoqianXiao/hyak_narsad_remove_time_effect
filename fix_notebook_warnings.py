#!/usr/bin/env python3
"""
Fix warnings in the NARSAD trial tracking analysis notebook
"""

import json

# Read the current notebook
with open('trial_tracking_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix the visualization cells to remove warnings
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'plt.tight_layout()' in str(cell['source']):
        # Replace the problematic lines
        source_lines = cell['source']
        new_source = []
        for line in source_lines:
            if 'plt.tight_layout()' in line:
                new_source.append('# Adjust layout to prevent tight_layout warning\n')
                new_source.append('plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)\n')
            elif 'plt.show()' in line:
                new_source.append('plt.show()\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

# Also add matplotlib backend configuration at the beginning
# Find the import cell and add backend configuration
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'import matplotlib.pyplot as plt' in str(cell['source']):
        source_lines = cell['source']
        new_source = []
        for line in source_lines:
            if 'import matplotlib.pyplot as plt' in line:
                new_source.append('import matplotlib.pyplot as plt\n')
                new_source.append('# Configure matplotlib for notebook display\n')
                new_source.append('plt.rcParams[\'figure.max_open_warning\'] = 0\n')
                new_source.append('plt.rcParams[\'figure.dpi\'] = 100\n')
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

# Write the fixed notebook
with open('trial_tracking_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Fixed notebook warnings!")
print("✅ Replaced tight_layout() with subplots_adjust()")
print("✅ Added matplotlib configuration for better notebook display")
