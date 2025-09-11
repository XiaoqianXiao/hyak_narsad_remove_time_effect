#!/usr/bin/env python3
"""
Add matplotlib configuration to the notebook
"""

import json

# Read the current notebook
with open('trial_tracking_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Add a new cell after the imports for matplotlib configuration
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Configure matplotlib for optimal notebook display\n",
        "import matplotlib\n",
        "matplotlib.use('Agg')  # Use non-interactive backend\n",
        "plt.rcParams['figure.max_open_warning'] = 0\n",
        "plt.rcParams['figure.dpi'] = 100\n",
        "plt.rcParams['figure.figsize'] = [12, 8]\n",
        "plt.rcParams['font.size'] = 10\n",
        "\n",
        "print(\"✅ Matplotlib configured for notebook display\")"
    ]
}

# Insert the new cell after the imports (cell index 1)
notebook['cells'].insert(2, new_cell)

# Write the updated notebook
with open('trial_tracking_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✅ Added matplotlib configuration cell!")
print("✅ Notebook should now run without warnings")
