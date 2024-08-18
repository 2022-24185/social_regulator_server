#!/bin/bash
# Only need to run this if you're using ES-HyperNEAT, 
# as it needs the dot command from Graphviz to visualize the network.

# Remember to make this script executable by running chmod +x setup.sh before running it.
PYTHON_VERSION="3.12.4"
PROJECT_NAME="social_mediator_server_env_test"

# Debugging: Print the Python version
echo "Using Python version: ${PYTHON_VERSION}"


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

# Install pyenv-virtualenv plugin
if ! brew list pyenv-virtualenv &> /dev/null; then
    echo "Installing pyenv-virtualenv..."
    brew install pyenv-virtualenv
fi

# Initialize pyenv-virtualenv and add it to the shell configuration
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
export PATH="$PYENV_ROOT/shims:${PATH}"

# Debugging: Print the PATH
echo "PATH: $PATH"

# Debugging: Check if pyenv is initialized
if pyenv --version &> /dev/null; then
    echo "pyenv is initialized."
else
    echo "pyenv is not initialized."
fi

# Add pyenv init to shell if it's not already there
if ! grep -q 'pyenv init -)' ~/.zshrc &> /dev/null; then
    echo 'Initializing pyenv virtualenv...'
    echo "shell is $SHELL"
    echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
    echo 'Refreshing terminal...'
    source ~/.zshrc
fi

# Install Python 3.8 using pyenv, if not already installed
if ! pyenv versions | grep -q ${PYTHON_VERSION}; then
    echo "Installing Python ${PYTHON_VERSION} using pyenv..."
    pyenv install ${PYTHON_VERSION}
fi

echo "PyEnv-VirtualEnv with Python ${PYTHON_VERSION} is set up."

# Create a virtual environment for the project with Python 3.12.4
echo "Checking for existing virtual environment..."
if ! pyenv virtualenvs | grep -q "${PROJECT_NAME}"; then
    echo "Creating virtual environment for ${PROJECT_NAME} with python version ${PYTHON_VERSION}..."
    pyenv virtualenv ${PYTHON_VERSION} ${PROJECT_NAME}
else
   echo "Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate ${PROJECT_NAME}

# Debugging: Check if the virtual environment is activated
if [[ "$(pyenv version-name)" == "${PROJECT_NAME}" ]]; then
    echo "Virtual environment ${PROJECT_NAME} is activated."
else
    echo "Failed to activate virtual environment ${PROJECT_NAME}."
    echo "Try restarting your terminal and running the script again."
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null
then
    echo "pip3 is not installed. Installing..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
fi

# Install the required Python packages
echo "Installing Python packages for manim..."
brew install py3cairo ffmpeg
brew install pango pkg-config scipy
echo "Installing the required Python packages..."
pip3 install -r requirements.txt
