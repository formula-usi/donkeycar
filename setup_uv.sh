#!/bin/bash
# DonkeyCar UV Environment Setup
# This script sets up UV environments for different platforms without requiring Python first

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo -e "${BOLD}$1${NC}"
}

# Function to print banner
print_banner() {
    echo "================================================================"
    echo "🏎️  DonkeyCar UV Environment Setup"
    echo "================================================================"
    echo ""
}

# Function to detect platform
detect_platform() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Check for Raspberry Pi
        if [[ -f /proc/cpuinfo ]] && grep -q "Raspberry Pi\|BCM" /proc/cpuinfo; then
            echo "pi"
        # Check for Jetson Nano
        elif [[ -f /proc/device-tree/model ]] && grep -qi "jetson" /proc/device-tree/model; then
            echo "nano"
        # Detect NVIDIA NGC container
        elif [[ -f /etc/nv_tegra_release ]] || \
           (grep -qi "NVIDIA" /etc/os-release 2>/dev/null && grep -qi "NGC" /etc/os-release 2>/dev/null); then
            echo "ngc"
            
        # Detect DGX Spark (ARM Neoverse + NVIDIA GPUs)
        elif [[ -f /etc/dgx-release ]] || grep -q 'DGX' /sys/class/dmi/id/product_name 2>/dev/null; then
            if [[ -f /.dockerenv ]]; then
                echo "ngc"
            else
                echo "spark"
            fi
        else
            echo "pc"
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "pc"
    else
        echo "pc"
    fi
}

# Function to check if UV is installed
check_uv() {
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
        print_status "UV is installed: $UV_VERSION"
        return 0
    else
        print_error "UV is not installed"
        return 1
    fi
}

# Function to install UV
install_uv() {
    print_info "Installing UV package manager (default behavior)..."
    
    if command -v curl &> /dev/null; then
        print_info "Installing UV using the official installer..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh; then
            # Add to current session PATH
            export PATH="$HOME/.local/bin:$PATH"
            # Add to shell profile for future sessions
            if [[ "$SHELL" == *"zsh"* ]] && [[ -f "$HOME/.zshrc" ]]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
            elif [[ "$SHELL" == *"bash"* ]] && [[ -f "$HOME/.bashrc" ]]; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            fi
            print_status "UV installed successfully"
            return 0
        else
            print_error "Failed to install UV using official installer"
        fi
    fi
    
    # Fallback to pip if available
    if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
        print_info "Trying to install UV using pip..."
        if pip install uv 2>/dev/null || pip3 install uv 2>/dev/null; then
            print_status "UV installed successfully via pip"
            return 0
        else
            print_error "Failed to install UV via pip"
        fi
    fi
    
    print_error "Could not install UV automatically"
    print_info "Please install UV manually:"
    print_info "  Visit: https://docs.astral.sh/uv/getting-started/installation/"
    return 1
}

# Function to create pyproject.toml
create_pyproject_toml() {
    print_info "Creating/Regenerating pyproject.toml (default behavior)..."
    
    cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "donkeycar"
dynamic = ["version"]
description = "Self driving library for python."
readme = "README.md"
requires-python = ">=3.11,<3.12"
license = {text = "MIT"}
authors = [
    {name = "Will Roscoe"},
    {name = "Adam Conway"},
    {name = "Tawn Kramer"},
]
keywords = ["selfdriving", "cars", "donkeycar", "diyrobocars"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
]

# Core dependencies that work across all platforms
dependencies = [
    "numpy",
    "pillow",
    "docopt",
    "tornado",
    "requests",
    "PrettyTable",
    "paho-mqtt",
    "simple_pid",
    "progress",
    "pyfiglet",
    "psutil",
    "pynmea2",
    "pyserial",
    "utm",
    "pandas",
    "pyyaml",
]

[project.urls]
Homepage = "https://github.com/autorope/donkeycar"
Repository = "https://github.com/autorope/donkeycar"
Issues = "https://github.com/autorope/donkeycar/issues"

[project.scripts]
donkey = "donkeycar.management.base:execute_from_command_line"

[project.optional-dependencies]
# Raspberry Pi specific dependencies
pi = [
    "picamera2",
    "Adafruit_PCA9685",
    "adafruit-circuitpython-ssd1306",
    "adafruit-circuitpython-rplidar",
    "RPi.GPIO",
    "tensorflow-aarch64==2.15.*",
    "opencv-contrib-python",
]

# Jetson Nano specific dependencies
nano = [
    "Adafruit_PCA9685",
    "adafruit-circuitpython-ssd1306",
    "adafruit-circuitpython-rplidar",
    "Jetson.GPIO",
    "numpy==1.23.*",
    "matplotlib==3.7.*",
    "kivy",
    "plotly",
    "pandas==2.0.*",
]

# PC (Linux/Windows) specific dependencies
pc = [
    "tensorflow[and-cuda]==2.15.*",
    "matplotlib",
    "kivy",
    "pandas",
    "plotly",
    "albumentations==1.3.1",
    "numpy<2",
    "opencv-python",
]

# Nvidia GPU Container specific dependencies
ngc = [
    "matplotlib",
    "kivy",
    "pandas",
    "plotly",
    "albumentations==1.3.1",
    "numpy<2",
    "opencv-python",
]

# macOS specific dependencies
macos = [
    "tensorflow-macos==2.15.*",
    "matplotlib",
    "kivy",
    "pandas",
    "plotly",
    "albumentations",
    "opencv-python",
    "gym==0.22.0",
]

# Gym Donkey Car simulator dependencies (installed separately)
gym-donkeycar = [
    "gym==0.22.0",
]

# Development dependencies
dev = [
    "pytest",
    "pytest-cov",
    "responses",
    "mypy",
    "black",
    "isort",
    "flake8",
]

# PyTorch dependencies (cross-platform)
torch = [
    "torch==2.1.*",
    "pytorch-lightning",
    "torchvision",
    "torchaudio",
    "fastai",
]

[tool.setuptools]
packages = ["donkeycar"]
include-package-data = true

[tool.setuptools.dynamic]
version = {attr = "donkeycar.__version__"}

[tool.setuptools.package-data]
"*" = ["*.html", "*.ini", "*.txt", "*.kv"]

# UV-specific configuration
[dependency-groups]
dev = [
    "pytest",
    "pytest-cov", 
    "responses",
    "mypy",
    "black",
    "isort",
    "flake8",
]
EOF

    print_status "Created/Regenerated pyproject.toml"
}

# Function to setup UV environment
setup_uv_environment() {
    local platform=$1
    local extras=${2:-""}
    local install_tensorrt=${3:-"auto"}
    
    print_info "Setting up UV environment for platform: $platform"
    
    # Build extras string
    local extras_str="$platform"
    if [[ -n "$extras" ]]; then
        extras_str="$platform,$extras"
    fi
    
    print_info "Installing donkeycar with extras: $extras_str"
    
    # Create virtual environment
    print_info "Creating virtual environment..."
    uv venv --system-site-packages .venv
    
    # Install global dependencies
    # print_info "Installing donkeycar with global dependencies"
    # if uv pip install -e ".[dependencies]"; then
    #     print_status "Global dependencies installed successfully"
    # else
    #     print_error "Failed to install global dependencies"
    #     return 1
    # fi

    # Install the package with only the specified platform extras
    print_info "Installing donkeycar with platform dependencies for: $platform"
    if uv pip install -e ".[$platform]"; then
        print_status "Platform dependencies installed successfully"
    else
        print_error "Failed to install platform dependencies"
        return 1
    fi
    
    # Install additional extras if specified
    if [[ -n "$extras" ]]; then
        IFS=',' read -ra EXTRA_ARRAY <<< "$extras"
        for extra in "${EXTRA_ARRAY[@]}"; do
            print_info "Installing extra: $extra"
            if uv pip install -e ".[$extra]"; then
                print_status "Extra '$extra' installed successfully"
            else
                print_warning "Failed to install extra: $extra"
            fi
        done
    fi
    
    # Install gym-donkeycar if gym is requested or if platform is macos
    if [[ "$extras_str" == *"gym-donkeycar"* ]] || [[ "$platform" == "macos" ]]; then
        if [[ -d "gym-donkeycar" ]]; then
            print_info "Installing local gym-donkeycar package..."
            if uv pip install -e ./gym-donkeycar; then
                print_status "gym-donkeycar installed successfully"
            else
                print_warning "Failed to install gym-donkeycar"
            fi
        else
            print_warning "gym-donkeycar directory not found, skipping installation"
        fi
    fi
    
    # Install TensorRT for PC platform (optional GPU optimization)
    if [[ "$platform" == "pc" ]] && [[ "$install_tensorrt" != "no" ]]; then
        print_info "Installing TensorRT for GPU optimization (this may take a while)..."
        # For TensorFlow 2.15 with CUDA 12.2, we need TensorRT compatible with CUDA 12
        # Note: TensorRT 10.x requires CUDA 13, so we skip for now
        # Users can install manually if they have compatible CUDA version
        print_warning "TensorRT auto-installation is currently not available for TensorFlow 2.15/CUDA 12"
        print_info "The TF-TRT warning can be safely ignored for PC environments"
        print_info "If you need TensorRT, consider upgrading to TensorFlow 2.16+ with CUDA 13"
    fi
    
    print_status "Successfully set up donkeycar environment for $platform"
    return 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -p, --platform PLATFORM    Force specific platform (pi, nano, pc, macos)"
    echo "  -e, --extras EXTRAS         Additional extras (dev, torch)"
    echo "  -t, --tensorrt              Install TensorRT for GPU optimization (PC only)"
    echo "  --no-tensorrt               Skip TensorRT installation"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Default Behavior:"
    echo "  * UV package manager will be installed if not present."
    echo "  * pyproject.toml will be created/regenerated."
    echo ""
    echo "Examples:"
    echo "  $0                          # Auto-detect platform and setup (default behavior)"
    echo "  $0 -p pi                    # Force Raspberry Pi setup"
    echo "  $0 -e dev,torch             # Include development and PyTorch extras"
    echo "  $0 -p nano -e dev           # Jetson Nano with dev tools"
    echo "  $0 -t                       # Setup with TensorRT (PC)"
    echo "  $0 --no-tensorrt            # Setup without TensorRT"
}

# Function to validate environment
validate_environment() {
    if [[ ! -d "donkeycar" ]]; then
        print_error "This doesn't appear to be a DonkeyCar project directory"
        print_info "Please run this script from the DonkeyCar project root"
        return 1
    fi
    return 0
}

# Parse command line arguments
PLATFORM=""
EXTRAS=""
INSTALL_TENSORRT="auto"

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -e|--extras)
            EXTRAS="$2"
            shift 2
            ;;
        -t|--tensorrt)
            INSTALL_TENSORRT="yes"
            shift
            ;;
        --no-tensorrt)
            INSTALL_TENSORRT="no"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_banner

# Validate environment
if ! validate_environment; then
    exit 1
fi

# Check/install UV (Now default if not found)
if ! check_uv; then
    if ! install_uv; then
        exit 1
    fi
fi

# Create pyproject.toml (Now default and always runs)
create_pyproject_toml

# Detect or use specified platform
if [[ -z "$PLATFORM" ]]; then
    PLATFORM=$(detect_platform)
    print_info "Auto-detected platform: $PLATFORM"
else
    print_info "Using specified platform: $PLATFORM"
fi

# Validate platform
case $PLATFORM in
    pi|nano|pc|macos|ngc|spark)
        ;;
    *)
        print_error "Invalid platform: $PLATFORM"
        print_info "Valid platforms: pi, nano, pc, macos"
        exit 1
        ;;
esac

# Show environment info
print_header "Environment Configuration:"
echo "  Platform: $PLATFORM"
echo "  Python: 3.11"
if [[ -n "$EXTRAS" ]]; then
    echo "  Extras: $EXTRAS"
fi
echo ""

# Setup environment
if setup_uv_environment "$PLATFORM" "$EXTRAS" "$INSTALL_TENSORRT"; then
    echo ""
    echo "================================================================"
    print_header "🎉 UV Environment Setup Complete!"
    echo "================================================================"
    echo ""
    print_status "Platform: $PLATFORM"
    if [[ -n "$EXTRAS" ]]; then
        print_status "Extras: $EXTRAS"
    fi
    echo ""
    print_header "Next steps:"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Create a car: donkey createcar --path ~/mycar"
    echo "  3. Start development!"
    echo ""
    print_info "Alternative - run commands with UV (no activation needed):"
    echo "  uv run python your_script.py"
    echo "  uv run donkey createcar --path ~/mycar"
    echo ""
    print_info "To add more packages:"
    echo "  uv add package-name"
    echo "  # or activate and use: pip install package-name"
else
    print_error "Environment setup failed"
    exit 1
fi
