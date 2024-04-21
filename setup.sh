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

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 is not installed. Installing..."
    case "$(uname -s)" in
        Darwin)
            brew install python3
            ;;
        Linux)
            sudo apt-get install python3
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            # Windows compatibility layer (Git Bash, etc.)
            echo "Please install Python 3 manually."
            ;;
        *)
            echo "Unknown operating system. Please install Python 3 manually."
            ;;
    esac
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip3 is not installed. Installing..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
fi

# Check if virtualenv is installed
if ! pip3 show virtualenv &> /dev/null
then
    echo "virtualenv is not installed. Installing..."
    pip3 install virtualenv
fi

# Create a virtual environment
echo "Checking for existing virtual environment..."
if [ -d "social_regulator_server_env" ]
then
    echo "Virtual environment already exists."
else
    echo "Creating a virtual environment..."
    python3 -m venv social_regulator_server_env
fi

# Install the required Python packages
pip3 install -r requirements.txt

# Activate the virtual environment
echo "Activating the virtual environment..."
source social_regulator_server_env/bin/activate