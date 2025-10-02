#!/bin/bash
# Verification script for DonkeyCar UV setup

echo "🔍 DonkeyCar Environment Verification"
echo "======================================"

# Check if virtual environment exists
if [[ -d ".venv" ]]; then
    echo "✅ Virtual environment found"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Check if activation works
if source .venv/bin/activate 2>/dev/null; then
    echo "✅ Virtual environment activation works"
else
    echo "❌ Virtual environment activation failed"
    exit 1
fi

# Test DonkeyCar import
echo -n "🧪 Testing DonkeyCar import... "
if python -c "import donkeycar; print('v' + donkeycar.__version__)" 2>/dev/null; then
    echo "✅ DonkeyCar import successful"
else
    echo "❌ DonkeyCar import failed"
    deactivate
    exit 1
fi

# Test basic dependencies
echo -n "🧪 Testing core dependencies... "
if python -c "import numpy, pandas, matplotlib; import cv2; print('Core deps OK')" 2>/dev/null; then
    echo "✅ Core dependencies working"
else
    echo "⚠️  Some core dependencies missing"
    # Debug: check what's failing
    echo "   Debugging individual imports:"
    python -c "import numpy; print('   ✅ numpy')" 2>/dev/null || echo "   ❌ numpy"
    python -c "import pandas; print('   ✅ pandas')" 2>/dev/null || echo "   ❌ pandas"  
    python -c "import matplotlib; print('   ✅ matplotlib')" 2>/dev/null || echo "   ❌ matplotlib"
    python -c "import cv2; print('   ✅ opencv')" 2>/dev/null || echo "   ❌ opencv"
fi

# Test platform-specific dependencies
echo -n "🧪 Testing platform dependencies... "
PLATFORM=$(uname -s)
if [[ "$PLATFORM" == "Darwin" ]]; then
    # Test macOS TensorFlow
    if python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)" 2>/dev/null; then
        echo "✅ TensorFlow-macOS working"
    else
        echo "⚠️  TensorFlow import failed (may need separate installation)"
    fi
elif [[ "$PLATFORM" == "Linux" ]]; then
    # Could add Linux-specific tests here
    echo "✅ Linux platform detected"
else
    echo "ℹ️  Platform: $PLATFORM"
fi

# Test donkey command
echo -n "🧪 Testing donkey command... "
if python -c "from donkeycar.management.base import execute_from_command_line; print('Donkey CLI available')" 2>/dev/null; then
    echo "✅ Donkey command available"
else
    echo "❌ Donkey command not working"
fi

echo ""
echo "🎉 Verification complete!"
echo ""
echo "💡 To use the environment:"
echo "   source .venv/bin/activate"
echo "   # or use: uv run <command>"

deactivate