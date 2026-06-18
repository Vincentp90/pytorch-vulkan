# Vulkan Build Instructions for PyTorch

## 1. Prerequisites

### 1.1 Vulkan SDK

**Windows:**
- Download and install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (latest stable version, 1.3.x recommended)
- Default install path: `C:\VulkanSDK\1.3.xxx\x86_64`
- The installer sets `VULKAN_SDK` environment variable automatically

**Linux (Ubuntu/Debian):**
```bash
# Using LunarG repository (recommended)
sudo apt-get install -y wget gnupg
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-trusty.list https://packages.lunarg.com/lunarg-vulkan-trusty.list
sudo apt-get update
sudo apt-get install vulkan-sdk

# Or using package manager (may be older version)
sudo apt-get install libvulkan-dev vulkan-tools
```

**Linux (RHEL/Fedora):**
```bash
sudo dnf install vulkan-devel vulkan-tools
```

**Android:**
- Vulkan is included in the NDK (no separate SDK needed)
- Set `ANDROID_NDK` CMake variable

### 1.2 glslc (GLSL Compiler)

`glslc` is bundled with the Vulkan SDK:

| Platform | Location |
|----------|----------|
| Windows | `%VULKAN_SDK%\bin\glslc.exe` |
| Linux (x86_64) | `$VULKAN_SDK/bin/glslc` |
| Android | `$ANDROID_NDK/shader-tools/<host>/glslc` |

Verify installation:
```bash
glslc --version
```

### 1.3 Build Tools

| Tool | Minimum Version | Notes |
|------|-----------------|-------|
| CMake | 3.27+ | Required by PyTorch |
| C++ Compiler | GCC 11.3+ / Clang 16+ / MSVC 2019+ | C++20 support required |
| Python | 3.9+ | For codegen scripts |
| Ninja | Latest | Recommended build generator |

```bash
# Install build tools (Ubuntu)
sudo apt-get install cmake ninja-build python3 python3-pip python3-venv

# Install build tools (Windows via winget)
winget install Kitware.CMake
winget install Ninja-build.Ninja
winget install Python.Python.3.12

# Install Python dependencies
pip install pyyaml
```

### 1.4 PyTorch Source

```bash
git clone https://github.com/pytorch/pytorch.git --recurse-submodules
cd pytorch
git submodule update --init --recursive
```

---

## 2. Build Configuration

### 2.1 CMake Options

| Option | Value | Description |
|--------|-------|-------------|
| `USE_VULKAN` | `ON` | Enable Vulkan backend |
| `USE_VULKAN_FP16_INFERENCE` | `ON`/`OFF` | Use fp16 inference (rgba16f textures) |
| `USE_VULKAN_RELAXED_PRECISION` | `ON`/`OFF` | Use `mediump` in GLSL shaders |
| `USE_CUDA` | `ON`/`OFF` | CUDA support (optional, for CPU↔Vulkan copy) |
| `USE_DISTRIBUTED` | `OFF` | Recommended to disable for Vulkan builds |
| `BUILD_PYTHON` | `ON`/`OFF` | Build Python bindings |
| `BUILD_TEST` | `ON`/`OFF` | Build Vulkan tests |
| `CMAKE_BUILD_TYPE` | `Release` / `RelWithDebInfo` | Build type |

### 2.2 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VULKAN_SDK` | Yes (non-Android) | Vulkan SDK install path |
| `PYTHONPATH` | Auto-set | Set by CMake to include pytorch root |

---

## 3. Build Commands

### 3.1 Windows (MSVC + Ninja)

```powershell
# Set environment
$env:VULKAN_SDK = "C:\VulkanSDK\1.3.290.0\x86_64"

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DUSE_VULKAN=ON ^
    -DUSE_VULKAN_FP16_INFERENCE=OFF ^
    -DUSE_VULKAN_RELAXED_PRECISION=OFF ^
    -DUSE_CUDA=OFF ^
    -DUSE_DISTRIBUTED=OFF ^
    -DBUILD_TEST=ON ^
    -DPython_EXECUTABLE=%CONDA_PREFIX%\python.exe

# Build
ninja -j$(nproc)
```

### 3.2 Linux (GCC + Ninja)

```bash
export VULKAN_SDK=/usr
mkdir -p build && cd build

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_VULKAN=ON \
    -DUSE_VULKAN_FP16_INFERENCE=OFF \
    -DUSE_VULKAN_RELAXED_PRECISION=OFF \
    -DUSE_CUDA=OFF \
    -DUSE_DISTRIBUTED=OFF \
    -DBUILD_TEST=ON

ninja -j$(nproc)
```

### 3.3 Linux (with system Vulkan SDK at non-standard path)

```bash
export VULKAN_SDK=/opt/vulkan-sdk/1.3.290
cmake .. -G Ninja \
    -DUSE_VULKAN=ON \
    -DUSE_CUDA=OFF \
    -DUSE_DISTRIBUTED=OFF
```

### 3.4 Android

```bash
mkdir -p build-android && cd build-android

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_VULKAN=ON \
    -DANDROID_NDK=/path/to/android-ndk-r26 \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF
```

---

## 4. Build Process: What Happens

When building with `USE_VULKAN=ON`, CMake executes these steps:

### 4.1 Dependency Resolution (`cmake/VulkanDependencies.cmake`)

```
find_package(Vulkan)
  → Locates Vulkan headers (Vulkan/include)
  → Locates Vulkan library (Vulkan/lib/Vulkan.lib or libvulkan.so)
  → Sets: Vulkan_INCLUDE_DIRS, Vulkan_LIBRARIES, Vulkan_INCLUDE_DIR
```

### 4.2 Shader Codegen (`cmake/VulkanCodegen.cmake`)

```
1. Find glslc compiler
   → Searches $VULKAN_SDK/bin or $ANDROID_NDK/shader-tools

2. Run gen_vulkan_spv.py
   → Scans aten/src/ATen/native/vulkan/glsl/ for .glsl files
   → For each .glsl: preprocess with YAML template vars, compile via glslc
   → Produces .spv (SPIR-V binary) files in build/vulkan/spv/

3. Generate spv.h / spv.cpp
   → Embeds all .spv binaries as C++ arrays
   → Registers shader metadata (tile_size, descriptor layouts, dispatch keys)
   → Output: build/aten/src/ATen/native/vulkan/spv.cpp
```

### 4.3 Compilation (`aten/src/ATen/CMakeLists.txt`)

```
vulkan_cpp          → aten/src/ATen/vulkan/*.cpp
native_vulkan_cpp   → aten/src/ATen/native/vulkan/*.cpp
                    → aten/src/ATen/native/vulkan/api/*.cpp
                    → aten/src/ATen/native/vulkan/impl/*.cpp
                    → aten/src/ATen/native/vulkan/ops/*.cpp
vulkan_generated_cpp→ build/aten/src/ATen/native/vulkan/spv.cpp  (codegen)

All compiled into libtorch_cpu (or libcaffe2) with -DUSE_VULKAN and -DUSE_VULKAN_API defines.
```

### 4.4 Shader Variants

The codegen system supports shader variant generation via YAML templates:

```yaml
# Example: aten/src/ATen/native/vulkan/glsl/binary_op.yaml
binary_op:
  parameter_names_with_default_values:
    OP: "ADD"
    DTYPE: "float"
    DIM: 2
  shader_variants:
    - NAME: binary_op_add_float_2d
      OP: "+"
      DTYPE: float
      DIM: 2
    - NAME: binary_op_add_half_2d
      OP: "+"
      DTYPE: half
      DIM: 2
  generate_variant_forall:
    OP:
      - VALUE: "+"
        SUFFIX: add
      - VALUE: "-"
        SUFFIX: sub
      - VALUE: "*"
        SUFFIX: mul
    DTYPE:
      - VALUE: float
      - VALUE: half
```

This generates: `binary_op_add_float_2d`, `binary_op_sub_float_2d`, `binary_op_add_half_2d`, etc.

---

## 5. Build Output

After a successful build, you get:

```
build/
├── lib/                          # Libraries
│   ├── torch_cpu.dll/.so/.dylib  # Main PyTorch library (includes Vulkan code)
│   └── ...
├── bin/                          # Tools
│   └── ...
└── vulkan/                       # Generated shader artifacts
    ├── spv/                      # Compiled SPIR-V binaries
    │   ├── binary_op_add_float_2d.spv
    │   ├── binary_op_mul_half_2d.spv
    │   └── ...
    └── ATen/native/vulkan/
        ├── spv.h                 # C++ header with shader metadata
        └── spv.cpp               # C++ source with embedded SPIR-V binaries
```

The Vulkan code is **statically linked** into `torch_cpu` — there is no separate `torch_vulkan` library.

---

## 6. Build Flags Reference

### 6.1 USE_VULKAN_FP16_INFERENCE

When `ON`:
- GLSL shaders use `rgba16f` (16-bit float) textures
- Reduces memory bandwidth by ~2x for floating-point ops
- May lose precision for extreme values
- Recommended for inference workloads

When `OFF` (default):
- GLSL shaders use `rgba32f` (32-bit float) textures
- Full precision
- Recommended for training or precision-sensitive ops

### 6.2 USE_VULKAN_RELAXED_PRECISION

When `ON`:
- GLSL shaders use `mediump` precision qualifiers
- Significant performance improvement on mobile GPUs
- May lose precision for large tensors or extreme values

When `OFF` (default):
- GLSL shaders use `highp` precision qualifiers
- Full precision on all devices

---

## 7. Troubleshooting

### 7.1 "USE_VULKAN glslc not found"

```
CMake Error: USE_VULKAN glslc not found
```

**Fix:** Ensure `VULKAN_SDK` is set and `glslc` is in the path:
```bash
# Windows
echo %VULKAN_SDK%
dir %VULKAN_SDK%\bin\glslc.exe

# Linux
echo $VULKAN_SDK
which glslc
```

If `glslc` is at a non-standard location, pass it explicitly:
```bash
cmake .. -DGLSLC_PATH=/path/to/glslc
```

### 7.2 "USE_VULKAN requires either Vulkan installed on system path or environment var VULKAN_SDK set"

```
CMake Error: USE_VULKAN requires either Vulkan installed on system path or environment var VULKAN_SDK set.
```

**Fix:** Install Vulkan SDK and set the environment variable before running CMake.

On Windows, the SDK installer should set `VULKAN_SDK` automatically. If not:
```powershell
# Add to environment
[Environment]::SetEnvironmentVariable("VULKAN_SDK", "C:\VulkanSDK\1.3.290.0\x86_64", "User")
# Or for current session only
$env:VULKAN_SDK = "C:\VulkanSDK\1.3.290.0\x86_64"
```

### 7.3 Shader compilation errors

```
CMake Error: Failed to gen spv.h and spv.cpp with precompiled shaders for Vulkan backend
```

**Fix:** 
1. Check glslc version — must support Vulkan 1.0+
2. Verify GLSL shaders have valid syntax
3. Run glslc manually on a shader to debug:
   ```bash
   glslc -fshader-stage=compute input.glsl -o output.spv --target-env=vulkan1.0
   ```

### 7.4 Build too slow

Vulkan shader compilation adds significant build time. Tips:
- Use `ccache` or `sccache` — SPIR-V compilation is cacheable
- For development, build with `USE_VULKAN=OFF` and only run Vulkan tests selectively
- On Windows, MSVC parallel compilation (`/MP`) helps

### 7.5 Runtime: "Vulkan not available"

```python
>>> import torch
>>> torch.device('vulkan')
RuntimeError: Vulkan not available
```

**Fix:**
1. Verify Vulkan drivers are installed (`vulkaninfo` on Linux, `vkcube` on Windows)
2. Check that the Vulkan backend was compiled in:
   ```python
   >>> torch.__config__.show()  # Look for USE_VULKAN
   ```
3. On Windows, ensure GPU drivers support Vulkan 1.0+

---

## 8. Running Tests

```bash
# After building with BUILD_TEST=ON:

# Run Vulkan API tests (via pytest)
python -m pytest aten/src/ATen/test/vulkan_api_test.cpp -v

# Or run the compiled test binary directly
./build/aten/src/ATen/test/vulkan_api_test
```

---

## 9. Clean Build

```bash
rm -rf build/
mkdir build && cd build
# Re-run cmake + ninja
```

Note: The `build/vulkan/spv/` directory contains cached SPIR-V files. Deleting it forces re-compilation of all shaders.

---

## 10. Integration with setup.py (setuptools build)

For the setuptools-based build (used for pip packages):

```bash
# Set environment before pip install
export USE_VULKAN=1
export USE_VULKAN_FP16_INFERENCE=0
export VULKAN_SDK=/path/to/vulkan/sdk

pip install .
```

The `setup.py` / `setup_helpers/cmake.py` will forward these flags to CMake. The Vulkan-specific options are handled in `cmake/EnvVarForwarding.cmake`.

---

## 11. Build Dependency Graph

```
CMakeLists.txt
├── cmake/VulkanDependencies.cmake     → find Vulkan SDK, set Vulkan_LIBS, Vulkan_INCLUDES
├── cmake/VulkanCodegen.cmake          → glslc → gen_vulkan_spv.py → spv.h, spv.cpp
└── aten/src/ATen/CMakeLists.txt
    ├── GLOB vulkan_cpp                → aten/src/ATen/vulkan/*.cpp
    ├── GLOB native_vulkan_cpp         → aten/src/ATen/native/vulkan/{api,impl,ops}/*.cpp
    ├── vulkan_generated_cpp (codegen) → build/.../spv.cpp
    └── all_cpu_cpp += vulkan_cpp + native_vulkan_cpp + vulkan_generated_cpp
        → compiled into torch_cpu with -DUSE_VULKAN -DUSE_VULKAN_API
```
