#!/usr/bin/env python3
"""
Fix matplotlib display in the notebook to show figures
"""

import json

# Read the current notebook
with open('trial_tracking_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix the matplotlib configuration cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'matplotlib.use(' in str(cell['source']):
        # Replace the matplotlib configuration
        cell['source'] = [
            "# Configure matplotlib for optimal notebook display\n",
            "import matplotlib\n",
            "import matplotlib.pyplot as plt\n",
            "# Use interactive backend for notebook display\n",
            "matplotlib.use('inline')\n",
            "plt.rcParams['figure.max_open_warning'] = 0\n",
            "plt.rcParams['figure.dpi'] = 100\n",
            "plt.rcParams['figure.figsize'] = [12, 8]\n",
            "plt.rcParams['font.size'] = 10\n",
            "\n",
            "print(\"✅ Matplotlib configured for notebook display\")\n"
        ]
        break

# Also fix the visualization cells to use proper display
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'plt.show()' in str(cell['source']):
        source_lines = cell['source']
        new_source = []
        for line in source_lines:
            if 'plt.show()' in line:
                new_source.append('plt.show()\n')
                new_source.append('plt.close()  # Close figure to free memory\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

# Write the fixed notebook
with open('trial_tracking_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Fixed matplotlib display!")
print("✅ Changed to 'inline' backend for notebook display")
print("✅ Added plt.close() to free memory")
print("✅ Figures should now display properly")
