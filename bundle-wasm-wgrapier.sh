#!/bin/bash

# Script to build wasm applications for web deployment
# Usage: ./build-wasm.sh <binary_name> [output_dir]
#
# Examples:
#   ./build-wasm.sh all_examples3
#   ./build-wasm.sh all_examples2 ./my-dist

set -e

# Check if binary name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <binary_name> [output_dir]"
    echo "Example: $0 all_examples3"
    exit 1
fi

BINARY_NAME=$1
OUTPUT_DIR=${2:-./dist}
TARGET_DIR="target/wasm32-unknown-unknown/release"

echo "Building $BINARY_NAME for wasm32-unknown-unknown..."

# Check if wasm32-unknown-unknown target is installed
if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
    echo "Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# Check if wasm-bindgen-cli is installed
if ! command -v wasm-bindgen &> /dev/null; then
    echo "wasm-bindgen-cli not found. Installing..."
    cargo install wasm-bindgen-cli
fi

# Build the wasm binary
echo "Building release binary..."
cargo build --release --bin "$BINARY_NAME" --target wasm32-unknown-unknown

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate JS bindings and web files
echo "Generating web bindings..."
wasm-bindgen --out-dir "$OUTPUT_DIR" --target web \
    "$TARGET_DIR/${BINARY_NAME}.wasm"

# Create index.html
echo "Creating index.html..."
cat > "$OUTPUT_DIR/index.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BINARY_TITLE</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #000;
        }
        canvas {
            width: 100%;
            height: 100vh;
            display: block;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-family: Arial, sans-serif;
            font-size: 24px;
        }
    </style>
</head>
<body>
    <div id="loading">Loading...</div>
    <script type="module">
        import init from './BINARY_FILE.js';

        // Prevent context menu on canvas right-click
        document.addEventListener('contextmenu', (e) => {
            if (e.target.tagName === 'CANVAS') {
                e.preventDefault();
            }
        });

        async function run() {
            try {
                await init();
                document.getElementById('loading').style.display = 'none';
            } catch (error) {
                console.error('Error loading wasm:', error);
                document.getElementById('loading').textContent = 'Error loading application';
            }
        }

        run();
    </script>
</body>
</html>
EOF

# Replace placeholders in index.html
sed -i.bak "s/BINARY_TITLE/$BINARY_NAME/g" "$OUTPUT_DIR/index.html"
sed -i.bak "s/BINARY_FILE/$BINARY_NAME/g" "$OUTPUT_DIR/index.html"
rm "$OUTPUT_DIR/index.html.bak"

echo ""
echo "✓ Build complete!"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To test locally, you can run:"
echo "  python3 -m http.server --directory $OUTPUT_DIR 8080"
echo "Then open http://localhost:8080 in your browser"
echo ""
echo "To deploy, upload the contents of '$OUTPUT_DIR' to your web server."
