#!/usr/bin/env python3
"""
Add Jupyter magic commands for proper matplotlib display
"""

import json

# Read the current notebook
with open('trial_tracking_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the matplotlib configuration cell and add magic commands
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'matplotlib.use(' in str(cell['source']):
        # Add magic commands at the beginning
        cell['source'] = [
            "# Jupyter magic commands for matplotlib display\n",
            "%matplotlib inline\n",
            "%config InlineBackend.figure_format = 'retina'\n",
            "\n",
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

# Write the updated notebook
with open('trial_tracking_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Added Jupyter magic commands!")
print("✅ Added %matplotlib inline for proper display")
print("✅ Added retina format for high-quality figures")
print("✅ Figures should now display correctly in the notebook")
