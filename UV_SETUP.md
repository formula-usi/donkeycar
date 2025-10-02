# UV Environment Setup for DonkeyCar

This guide explains how to set up a UV environment for DonkeyCar that supports multiple platforms (Raspberry Pi, Jetson Nano, PC, macOS) with platform-specific dependencies.

## What is UV?

[UV](https://docs.astral.sh/uv/) is a fast Python package manager and project manager written in Rust. It's designed to be a drop-in replacement for pip, pip-tools, pipx, poetry, pyenv, and virtualenv, all in one tool.

## Quick Start

### 1. Run the setup script

The setup script will automatically install UV (if needed) and set up the environment:

```bash
# Auto-detect platform and set up environment
./setup_uv.sh

# Or install UV automatically if missing
./setup_uv.sh --install-uv
```

### 2. Activate and use the environment

```bash
# Activate the environment (standard method)
source .venv/bin/activate

# Or run commands directly with UV (recommended)
uv run donkey createcar --path ~/mycar
uv run python my_script.py
```

## Manual UV Installation (Optional)

If you prefer to install UV manually first:

```bash
# On Unix-like systems (macOS, Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip (if you already have Python)
pip install uv
```

## Platform-Specific Setup

### Raspberry Pi

```bash
./setup_uv.sh --platform pi
```

**Includes:**
- `picamera2` for camera support
- `tensorflow-aarch64` optimized for ARM64
- `RPi.GPIO` for GPIO control
- Hardware-specific libraries (PCA9685, SSD1306, etc.)

### Jetson Nano

```bash
./setup_uv.sh --platform nano
```

**Includes:**
- `Jetson.GPIO` for Jetson GPIO control
- CUDA-compatible versions of libraries
- Optimized NumPy, Matplotlib, and Pandas versions

### PC (Linux/Windows)

```bash
./setup_uv.sh --platform pc
```

**Includes:**
- Standard `tensorflow` for x86/x64
- Full OpenCV support
- Development tools (Kivy, Albumentations)

### macOS

```bash
./setup_uv.sh --platform macos
```

**Includes:**
- `tensorflow-macos` with Metal acceleration
- macOS-optimized packages
- Native Apple Silicon support

## Additional Extras

### Development Tools

```bash
./setup_uv.sh --extras dev
```

**Includes:**
- `pytest` for testing
- `black` for code formatting
- `mypy` for type checking
- `flake8` for linting

### PyTorch Support

```bash
./setup_uv.sh --extras torch
```

**Includes:**
- PyTorch with GPU support
- PyTorch Lightning
- FastAI framework

### Combining Extras

```bash
# Platform + development tools
./setup_uv.sh --platform pi --extras dev

# Platform + PyTorch + development
./setup_uv.sh --platform pc --extras dev,torch
```

## Advanced Usage

### Create Platform Configurations

Generate platform-specific configuration files:

```bash
# The script automatically creates pyproject.toml
# Platform-specific configs are in .uv/ directory
```

### Manual UV Commands

Once the environment is set up, you can use UV directly:

```bash
# Add a new package
uv add opencv-python

# Remove a package
uv remove numpy

# Update all packages
uv sync --upgrade

# Install from requirements
uv pip install -r requirements.txt

# Show installed packages
uv pip list

# Create a lock file
uv lock
```

## Environment Management

### Activate Environment

```bash
# Activate shell with environment (standard method)
source .venv/bin/activate

# Run single command in environment (UV method)
uv run python script.py
uv run donkey createcar --path ~/mycar
```

### Environment Information

```bash
# Show environment details
uv python list

# Show project information  
uv info

# Show dependency tree
uv tree
```

## Platform-Specific Notes

### Raspberry Pi

- Uses piwheels.org for pre-compiled ARM packages
- TensorFlow Lite is recommended for better performance
- GPU acceleration not available

### Jetson Nano

- Requires CUDA toolkit installation
- Limited memory - consider swap file
- Some packages may need compilation

### PC/macOS

- Full GPU acceleration available
- All features supported
- Fastest setup and execution

## Troubleshooting

### UV Not Found

```bash
# Add UV to PATH (if installed via curl)
export PATH="$HOME/.cargo/bin:$PATH"

# Or install via pip
pip install uv
```

### Platform Detection Issues

Force platform manually:

```bash
python setup_uv_env.py --platform pi  # Force Raspberry Pi
python setup_uv_env.py --platform nano # Force Jetson Nano
```

### Dependency Conflicts

```bash
# Reset environment
rm uv.lock
python setup_uv_env.py --platform <your_platform>

# Or use platform-specific config
uv sync --config-file .uv/pi.toml
```

### Import Errors

Make sure you're in the UV environment:

```bash
# Check if in environment
which python

# Activate if needed
uv shell

# Or run with uv
uv run python -c "import tensorflow; print(tensorflow.__version__)"
```

## Migration from pip/conda

### From requirements.txt

```bash
# Install existing requirements
uv pip install -r requirements.txt

# Convert to pyproject.toml
uv add $(cat requirements.txt)
```

### From conda environment

```bash
# Export conda environment
conda env export > environment.yml

# Install packages with UV
uv add numpy pandas tensorflow  # etc.
```

## Performance Benefits

UV offers significant performance improvements:

- **10-100x faster** package installation
- **Faster dependency resolution**
- **Smaller disk usage** with package caching
- **Reproducible builds** with lock files

## File Structure

After setup, your project will have:

```
donkeycar/
├── pyproject.toml          # Project configuration (auto-created)
├── uv.lock                 # Lock file (auto-generated)
├── setup_uv.sh             # Setup script
├── verify_setup.sh         # Verification script
├── .uv/                    # Platform configurations
│   ├── pi.toml
│   ├── nano.toml
│   ├── pc.toml
│   └── macos.toml
└── .venv/                  # Virtual environment (auto-created)
```

## Quick Verification

After setup, run the verification script:

```bash
./verify_setup.sh
```

This will test that all components are working correctly.

## Best Practices

1. **Always use lock files** - Commit `uv.lock` for reproducible builds
2. **Platform-specific extras** - Use appropriate platform extras
3. **Pin critical dependencies** - Specify versions for stability
4. **Use dry-run** - Test changes before applying
5. **Regular updates** - Keep dependencies current with `uv sync --upgrade`

## Getting Help

- **UV Documentation**: https://docs.astral.sh/uv/
- **DonkeyCar Documentation**: http://docs.donkeycar.com/
- **GitHub Issues**: Report platform-specific issues

## Environment Activation Methods

UV creates standard Python virtual environments, so you have multiple options:

### Method 1: Standard Virtual Environment (Recommended for development)
```bash
# Activate the environment
source .venv/bin/activate

# Now you can use python/pip/donkey directly
python script.py
pip install additional-package
donkey createcar --path ~/mycar

# Deactivate when done
deactivate
```

### Method 2: UV Run (Recommended for scripts)
```bash
# Run commands without activating
uv run python script.py
uv run donkey createcar --path ~/mycar
uv run pip install additional-package
```

## Example Workflows

### Raspberry Pi Setup

```bash
# On development machine - test the setup
./setup_uv.sh --platform pi

# Transfer to Raspberry Pi and run
./setup_uv.sh --platform pi --install-uv

# Create car
source .venv/bin/activate  # or use: uv run donkey createcar --path ~/mycar
donkey createcar --path ~/mycar
cd ~/mycar
python manage.py drive  # or use: uv run python manage.py drive
```

### Development Setup

```bash
# Full development environment
./setup_uv.sh --platform pc --extras dev,torch

# Activate environment
source .venv/bin/activate

# Run tests
pytest  # or use: uv run pytest

# Format code
black donkeycar/  # or use: uv run black donkeycar/

# Type checking
mypy donkeycar/  # or use: uv run mypy donkeycar/
```

### Cross-Platform Development

```bash
# Lock dependencies on development machine
uv lock

# Deploy to different platforms
git clone <repo>
python setup_uv_env.py --platform pi  # Auto-uses locked versions
```