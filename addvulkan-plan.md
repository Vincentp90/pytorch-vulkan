# Plan: Adding Vulkan Device Support to PyTorch

## 1. Background & Current State

This repository (pytorch-vulkan) already contains a **partial Vulkan backend scaffolding** for PyTorch:

- `DeviceType::Vulkan = 10` is registered in `torch/headeronly/core/DeviceType.h`
- `VulkanImplInterface` and `VulkanImplRegistrar` exist in `aten/src/ATen/vulkan/Context.h/.cpp`
- A stub `VulkanGuardImpl` exists in `aten/src/ATen/native/vulkan/VulkanGuardImpl.cpp` (returns 1 device, no-op stream management)
- `vTensor` / `vTensorStorage` wraps Vulkan memory (VkImage, VkBuffer) with barrier tracking
- Vulkan API layer (`Context`, `Pipeline`, `Shader`) manages devices, queues, descriptor pools
- Several ops are partially implemented: `BinaryOp.cpp` (compute job submission with GLSL shaders), `Convolution.h` (Conv2dPackedContext), `Linear.cpp`, `Layernorm.cpp`, `Pooling.cpp`, `UnaryOp.cpp`, `ReduceOp.cpp`

The Vulkan backend is **not yet wired into PyTorch's dispatch system** — ops still run on CPU/CUDA. The goal is to complete the backend so that `torch.device('vulkan')` creates tensors on Vulkan and ops dispatch to Vulkan kernels.

---

## 2. Architecture Overview: CUDA as Reference

CUDA backend structure (the template for Vulkan):

```
c10/cuda/                        # Core infrastructure
  - CUDAGuardImpl.h/.cpp         # DeviceGuardImplInterface implementation
  - CUDACachingAllocator.h/.cpp  # Memory allocator with pooling
  - CUDAStream.h                 # Stream management (default + worker pools)
  - CUDAFunctions.h              # CUDA runtime API wrappers (throw on error)
  - CPUGeneratorImpl.h          # (CUDA-specific generator)

aten/src/ATen/cuda/              # ATen CUDA backend
  - CUDAContext.h/.cpp           # Device properties, stream getters
  - CUDAException.h              # Error handling
  - Copy.cu                      # Cross-device copy operations
  - Blank.cu                     # Tensor blank/zero operations
  - Fill.cu                      # Fill operations

aten/src/ATen/native/cuda/       # Native CUDA operators
  - *.cu                         # Per-operator CUDA kernels
  - BinaryOps.cu, UnaryOps.cu
  - ReductionOps.cu, etc.
```

Vulkan needs an equivalent structure.

---

## 3. Implementation Phases

### Phase 1: Core Infrastructure (C10 Layer)

**Goal:** Wire Vulkan into PyTorch's device management system.

| # | File | Status | Work |
|---|------|--------|------|
| 1.1 | `aten/src/ATen/native/vulkan/VulkanGuardImpl.cpp` | Stub | Implement full `DeviceGuardImplInterface`: `getDevice()`, `setDevice()`, `exchangeDevice()`, `getDeviceCount()`, `getDeviceProperties()`, `getStream()`, `setStream()`, `lazyInitStream()` |
| 1.2 | `aten/src/ATen/native/vulkan/VulkanStream.h` | Missing | Create stream abstraction wrapping Vulkan `VkQueue`. Implement `VulkanStreamManager` with a default queue and worker queue pool (similar to CUDA's 32 low-priority + high-priority streams). |
| 1.3 | `aten/src/ATen/native/vulkan/VulkanCachingAllocator.h/.cpp` | Missing | Implement a Vulkan memory caching allocator. Vulkan uses `VkDeviceMemory` — pool allocations by size classes (like CUDA's async allocator). Track freed blocks, reuse allocations. |
| 1.4 | `aten/src/ATen/native/vulkan/VulkanExceptions.h` | Missing | Error-checking macros for Vulkan API calls (VK_CHECK macro that throws on `VkResult != VK_SUCCESS`). |
| 1.5 | `aten/src/ATen/native/vulkan/VulkanContext.h/.cpp` | Partial | `VulkanContext` needs: per-device properties (max image dims, memory limits, queue family info), stream accessor, device initialization. |
| 1.6 | `aten/src/ATen/native/vulkan/Register.cpp` | Partial | Ensure `VulkanImplInterface` is properly registered via `VulkanImplRegistrar`. Verify the singleton pattern works. |

**Key design decisions:**
- Vulkan uses command queues (not streams like CUDA). Map `at::Stream` → `VkQueue` (default queue family, submit queue).
- Vulkan has no native event primitive like CUDA events. Implement synchronization via fence objects or semaphore chains.
- Vulkan memory is explicitly managed — the caching allocator is **essential** for performance.

### Phase 2: Tensor & Memory Operations

**Goal:** Enable tensor creation, memory management, and data transfer on Vulkan.

| # | File | Status | Work |
|---|------|--------|------|
| 2.1 | `aten/src/ATen/native/vulkan/api/Tensor.h` | Partial | Extend `vTensor` with: allocation/deallocation hooks, memory type selection (DEVICE_LOCAL vs HOST_VISIBLE), proper refcounting. |
| 2.2 | `aten/src/ATen/native/vulkan/ops/Empty.cpp` | Missing | Implement `empty()` — allocate VkBuffer/VkImage with appropriate dimensions and format. |
| 2.3 | `aten/src/ATen/native/vulkan/ops/Blank.cpp` | Missing | Zero-fill a tensor (dispatch a compute shader that writes zeros). |
| 2.4 | `aten/src/ATen/native/vulkan/ops/Fill.cpp` | Missing | Fill tensor with a scalar value (compute shader with constant). |
| 2.5 | `aten/src/ATen/native/vulkan/ops/Copy.cpp` | Missing | Cross-device copy: CPU→Vulkan, Vulkan→CPU, Vulkan→CUDA (if CUDA is available). Use `vkCmdCopyBufferToImage` / `vkCmdCopyImageToBuffer` with host-mapped memory or staging buffers. |
| 2.6 | `aten/src/ATen/native/vulkan/ops/Clone.cpp` | Missing | Clone a tensor (allocate new, copy data). |
| 2.7 | `aten/src/ATen/native/vulkan/ops/Resize.cpp` | Missing | Resize tensor (realloc + copy if needed). |

**Copy strategy (CPU ↔ Vulkan):**
```
CPU → Vulkan:  Host buffer (HOST_VISIBLE) → vkCmdCopyBufferToImage → device buffer (DEVICE_LOCAL)
Vulkan → CPU:  device buffer → vkCmdCopyImageToBuffer → host buffer → map/unmap
```
Use staging buffers for optimal transfer. Sync with fences.

### Phase 3: Operator Implementation (Native Vulkan Kernels)

**Goal:** Implement all major operators as Vulkan compute shaders + host glue code.

This is the largest phase. Use the existing partial implementations as a template:

```
aten/src/ATen/native/vulkan/ops/
├── BinaryOp.cpp       # Element-wise: add, sub, mul, div, etc. (GLSL compute shader)
├── UnaryOp.cpp        # Element-wise: sin, cos, exp, log, abs, etc.
├── ReduceOp.cpp       # Reductions: sum, mean, max, min, prod (with shared memory)
├── Convolution.h      # Conv2d context — needs full kernel implementation
├── Linear.cpp         # Linear/GEMM — needs matrix multiply shader (or call Vulkan compute)
├── Layernorm.cpp      # Layer normalization
├── Pooling.cpp        # MaxPool, AvgPool
├── Embedding.cpp      # Lookup table
├── Dropout.cpp        # Random dropout
├── Softmax.cpp        # Softmax (reduce + broadcast)
├── Activation.cpp     # ReLU, sigmoid, tanh, gelu, silu
├── Indexing.cpp       # index, index_select, gather, scatter
├── ResizeOp.cpp       # reshape, view, transpose, permute (no-copy metadata)
├── Pad.cpp            # Padding
├── Upsample.cpp       # Interpolation (bilinear, nearest)
└── MathOp.cpp         # matmul, bmm, mm (GEMM via compute shader or external library)
```

**Shader pattern (from existing BinaryOp.cpp):**
```cpp
// Host side
at::Tensor add_cpu(const at::Tensor& self, const at::Tensor& other) {
    auto result = at::empty_like(self);
    auto ctx = at::globalContext();
    auto stream = at::getVulkanStream(ctx, self.device());
    
    auto* self_v = api::vTensor::get(self);
    auto* other_v = api::vTensor::get(other);
    auto* result_v = api::vTensor::get(result);
    
    submit_compute_job(ctx, stream, [self_v, other_v, result_v]() {
        // Build dispatch key, set descriptors, record compute command
    });
    
    return result;
}
```

**GLSL shader pattern:**
```glsl
#version 450
layout(local_size_x_id = 0) in;
layout(set = 0, binding = 0) buffer InputA { float a[]; };
layout(set = 0, binding = 1) buffer InputB { float b[]; };
layout(set = 0, binding = 2) buffer Output { float o[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    o[i] = a[i] + b[i];  // element-wise add
}
```

### Phase 4: Dispatch Integration

**Goal:** Make PyTorch dispatch ops to Vulkan when `device='vulkan'`.

| # | File | Status | Work |
|---|------|--------|------|
| 4.1 | `aten/src/ATen/native/vulkan/Register.cpp` | Partial | Register `VulkanImplInterface` so that `AT_DISPATCH` macros route Vulkan tensors to Vulkan implementations. |
| 4.2 | `native_functions.yaml` | Needs change | Add `vulkan` dispatch key entries for operators that have Vulkan implementations. |
| 4.3 | `aten/src/ATen/native/` | Needs change | For each operator with Vulkan impl, add `vulkan:` entry in YAML so codegen generates the dispatch path. |
| 4.4 | `c10/util/Device.h` | Verify | Ensure `vulkan` device type is recognized in `at::Device` parsing. |
| 4.5 | `torch/python/` | Verify | Ensure Python-side `torch.device('vulkan')` works. May need registration in `torch/csrc/utils/` |

**Dispatch chain:**
```
Tensor (device=vulkan) 
  → AT_DISPATCH_FLOATING_TYPES_AND_HALF (in native ops)
  → selects impl by dispatch key
  → Vulkan backend impl (our code)
```

### Phase 5: Autograd & Python Integration

**Goal:** Enable training (backward pass) and Python API.

| # | File | Status | Work |
|---|------|--------|------|
| 5.1 | `aten/src/ATen/native/vulkan/autograd/` | Missing | Create autograd functions for Vulkan ops. Each forward op needs a backward kernel or a way to call the CPU/CUDA autograd. |
| 5.2 | `torch/csrc/vulkan/` | Missing | Python bindings: `torch.vulkan` module, device selection, version info. |
| 5.3 | `torch/csrc/` | Needs change | Register Vulkan in Python's device type enum. |
| 5.4 | `aten/src/ATen/native/vulkan/ops/` | Needs change | Implement `grad()` for each autograd function. |

### Phase 6: Testing & Validation

| # | File | Status | Work |
|---|------|--------|------|
| 6.1 | `test/vulkan/` | Missing | Create test suite: correctness against CPU reference, memory leak detection, shader compilation validation. |
| 6.2 | `test/` | Needs change | Add Vulkan to existing test matrix (run CPU tests on Vulkan device). |
| 6.3 | CI/CD | Needs change | Add Vulkan GPU to CI (or mock validation layers). |

---

## 4. CUDA → Vulkan Mapping Reference

| CUDA Concept | Vulkan Equivalent | Notes |
|-------------|-------------------|-------|
| `cudaStream_t` | `VkQueue` (submit queue) | Vulkan queues are less flexible — no priority, no concurrent execution within same family |
| `cudaEvent_t` | `VkFence` / `VkSemaphore` | Fences for host synchronization, semaphores for queue-to-queue |
| `cudaMalloc` / `cudaFree` | `vkAllocateMemory` / `vkFreeMemory` | Must implement caching allocator for performance |
| `cudaMemcpy` | `vkCmdCopyBufferToImage` / `vkCmdCopyImageToBuffer` | Use staging buffers for optimal transfers |
| `__global__` kernel | GLSL `compute` shader | Compiled via `vkCreateShaderModule` + `vkCreatePipeline` |
| `cudaGetDeviceProperties` | `vkGetPhysicalDeviceProperties` / `vkGetPhysicalDeviceMemoryProperties` | |
| `CUDACachingAllocator` | Custom `VulkanCachingAllocator` | Pool by memory type + size class |
| `TensorIterator` | Same (CPU-side) | `TensorIterator` is CPU-side infrastructure — it works for all devices. Use it to plan the operation, then submit to Vulkan. |
| `at::Scalar` | Same | Scalar handling is CPU-side, works for all devices |
| `AT_DISPATCH_*` macros | Same | Use existing dispatch macros, add `kVulkan` to the list |

---

## 5. Priority & Effort Estimate

| Phase | Priority | Effort | Description |
|-------|----------|--------|-------------|
| 1. Core Infrastructure | P0 — Must have | Medium | Without this, Vulkan tensors can't be created or managed |
| 2. Tensor & Memory Ops | P0 — Must have | Medium | Essential for any meaningful computation |
| 3. Operator Kernels | P1 — High value | Large | The bulk of the work. Start with common ops (add, mul, matmul, conv, linear) |
| 4. Dispatch Integration | P0 — Must have | Medium | Ties everything together |
| 5. Autograd & Python | P2 — Important later | Large | Needed for training, not inference |
| 6. Testing | P1 — High value | Medium | Critical for correctness |

---

## 6. Immediate Next Steps (Sprint 1)

1. **Implement `VulkanGuardImpl`** — Full `DeviceGuardImplInterface` with device/stream management
2. **Implement `VulkanStreamManager`** — Map `at::Stream` to `VkQueue`
3. **Implement `VulkanCachingAllocator`** — Basic pool allocator for `VkDeviceMemory`
4. **Implement `empty()` and `copy()` ops** — Enable tensor creation and CPU↔Vulkan transfer
5. **Wire up dispatch** — Register Vulkan so `torch.device('vulkan')` creates vulkan tensors

After these 5 steps, you can create Vulkan tensors and move data between CPU and Vulkan, but most ops will fall back to CPU. Then proceed with Phase 3 (operator kernels).

---

## 7. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Vulkan shaders are slower than CUDA for some ops | Start with ops where Vulkan excels (image ops, parallel element-wise). Use CPU fallback for complex ops. |
| No mature Vulkan compute library equivalent to cuBLAS/cuDNN | Implement GEMM manually or integrate with a Vulkan compute library (e.g., Compute Library, or hand-written block GEMM shaders) |
| Memory fragmentation | Caching allocator with size-class pooling (like CUDA's async allocator) |
| Shader compilation latency | Pre-compile commonly used shaders at init time; cache compiled pipelines |
| Synchronization overhead | Batch commands where possible; minimize fence usage; use semaphores for queue sync |

---

## 8. File Structure (Proposed)

```
aten/src/ATen/native/vulkan/
├── Context.h/.cpp           # VulkanImplInterface, global context
├── GuardImpl.cpp            # DeviceGuardImplInterface (update existing)
├── Stream.h/.cpp            # NEW: Vulkan stream/queue management
├── CachingAllocator.h/.cpp  # NEW: Memory pool allocator
├── Exceptions.h             # NEW: VK_CHECK macros
├── Tensor.h                 # vTensor, vTensorStorage (update existing)
├── api/
│   ├── Context.h/.cpp       # Vulkan API layer (update existing)
│   ├── Tensor.h             # vTensor API (update existing)
│   ├── Pipeline.h/.cpp      # Shader/pipeline management (update existing)
│   └── Shader.h             # GLSL shader loading (update existing)
├── ops/
│   ├── Register.cpp         # Op registration (update existing)
│   ├── Empty.cpp            # NEW: tensor allocation
│   ├── Blank.cpp            # NEW: zero-fill
│   ├── Fill.cpp             # NEW: scalar fill
│   ├── Copy.cpp             # NEW: CPU↔Vulkan, Vulkan↔Vulkan copy
│   ├── Clone.cpp            # NEW: tensor clone
│   ├── Resize.cpp           # NEW: resize
│   ├── BinaryOp.cpp         # Update existing
│   ├── UnaryOp.cpp          # Update existing
│   ├── ReduceOp.cpp         # Update existing
│   ├── Convolution.h        # Update existing
│   ├── Linear.cpp           # Update existing
│   ├── Layernorm.cpp        # Update existing
│   ├── Pooling.cpp          # Update existing
│   ├── Activation.cpp       # NEW: ReLU, sigmoid, etc.
│   ├── MathOp.cpp           # NEW: matmul, bmm
│   ├── Indexing.cpp         # NEW: gather, scatter, index
│   └── Pad.cpp              # NEW: padding
└── autograd/
    └── (future)             # Backward pass implementations

torch/
└── csrc/
    └── vulkan/
        └── (future)         # Python bindings
```
