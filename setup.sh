#!/bin/bash
# Only need to run this if you're using ES-HyperNEAT, 
# as it needs the dot command from Graphviz to visualize the network.

# Remember to make this script executable by running chmod +x setup.sh before running it.

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Ubuntu
    sudo apt-get update
    sudo apt-get install graphviz
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac OSX
    brew install graphviz
else
    echo "Your OS is not supported by this script. Please install Graphviz manually."
fi

# Create a new virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Install the required Python packages
pip3 install -r requirements.txt