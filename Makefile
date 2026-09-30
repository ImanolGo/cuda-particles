# =============================================================================
#  CUDA from Scratch — build file
# =============================================================================
#
#  Everyday commands
#  -----------------
#     make                  build every example + the particle system
#     make 01_hello         build one example  (01_hello .. 06_interop)
#     make run-01_hello     build & run it
#     make particles        build the final project
#     make run-particles    build & run the final project
#     make doctor           print toolkit / GPU info
#     make clean
#
#  Tunables (override on the command line, e.g. `make ARCH=sm_75`)
#  ------------------------------------------------------------------
#     ARCH=native           GPU architecture to compile for.
#                           `native` asks nvcc to detect your card.
#     UNSUPPORTED=1         add -allow-unsupported-compiler.
#                           Needed if nvcc rejects your (very new) host GCC.
#     NVCC=/path/to/nvcc    use a specific toolkit, e.g. /opt/cuda-12.6/bin/nvcc
# =============================================================================

NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /opt/cuda/bin/nvcc)
ARCH      ?= native
BUILD     ?= build
GL_LIBS   := $(shell pkg-config --libs glfw3 gl 2>/dev/null || echo -lglfw -lGL)

NVCCFLAGS := -O2 -std=c++17 -lineinfo -arch=$(ARCH) -Xcompiler -Wall

ifdef UNSUPPORTED
NVCCFLAGS += -allow-unsupported-compiler
endif

EXAMPLES  := 01_hello 02_vector_add 03_index_math 04_shared_memory \
             05_timing_streams 06_interop

PARTICLE_SRC := $(wildcard particles/src/*.cu)
PARTICLE_HDR := $(wildcard particles/src/*.cuh) $(wildcard particles/src/*.h)

# -----------------------------------------------------------------------------
#  Default goal
# -----------------------------------------------------------------------------
all: $(addprefix $(BUILD)/,$(EXAMPLES)) $(BUILD)/particles

# -----------------------------------------------------------------------------
#  Examples:  build/<name>  from  examples/<name>/main.cu
#  (GL is linked into every example — harmless for the CPU-only ones and it
#   keeps this Makefile free of special cases.)
# -----------------------------------------------------------------------------
$(BUILD)/%: examples/%/main.cu common/cuda_check.h | $(BUILD)
	@echo "  NVCC  $<"
	@$(NVCC) $(NVCCFLAGS) -Icommon -o $@ $< $(GL_LIBS)

# Convenience aliases so `make 01_hello` works as well as `make run-01_hello`.
$(EXAMPLES): %: $(BUILD)/%

# -----------------------------------------------------------------------------
#  Final project
# -----------------------------------------------------------------------------
$(BUILD)/particles: $(PARTICLE_SRC) $(PARTICLE_HDR) common/cuda_check.h | $(BUILD)
	@echo "  NVCC  particles ($(words $(PARTICLE_SRC)) translation units)"
	@$(NVCC) $(NVCCFLAGS) -Icommon -Iparticles/src -o $@ $(PARTICLE_SRC) $(GL_LIBS)

particles: $(BUILD)/particles

# -----------------------------------------------------------------------------
#  Run / inspect / clean
# -----------------------------------------------------------------------------
#  On hybrid-graphics laptops (Optimus) the OpenGL context is created on the
#  integrated GPU by default, and CUDA/OpenGL interop is impossible there.
#  `prime-run` (from nvidia-prime) moves the context to the NVIDIA GPU, so we
#  use it automatically when it is installed. Disable with `make run-X RUN=`.
RUN ?= $(shell command -v prime-run 2>/dev/null)

run-%: $(BUILD)/%
	@echo "▶ $(if $(RUN),$(RUN) ,)$(BUILD)/$*"
	@$(RUN) ./$(BUILD)/$*

doctor:
	@echo "nvcc        : $(NVCC)"
	@$(NVCC) --version || echo "  (nvcc not found — see docs/00_setup.md)"
	@echo
	@echo "GPU         :"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null \
		|| echo "  (nvidia-smi unavailable)"

$(BUILD):
	@mkdir -p $@

clean:
	@rm -rf $(BUILD)

.PHONY: all clean doctor particles $(EXAMPLES)
