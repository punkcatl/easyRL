# LonMPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an independent longitudinal acceleration MPC module (`LonMPC`) using a triple-integrator model and HPIPM solver.

**Architecture:** The module lives as a standalone static library under `src/library/lon_mpc_lib/`. It depends only on HPIPM (and its dependency BLASFEO) which are added as thirdparty submodules. The module exposes a single `LonMPC` class with `Init/Update/Reset` interface.

**Tech Stack:** C++17, HPIPM (OCP QP solver), BLASFEO (linear algebra backend), CMake

---

## File Structure

```
src/library/lon_mpc_lib/
├── CMakeLists.txt
├── include/
│   ├── lon_mpc.h              # Public API: LonMPC class, LonMPCConfig, LonMPCInput, LonMPCOutput
│   ├── lon_mpc_model.h        # ModelBuilder: Ad/Bd computation, gradient/cost assembly
│   └── lon_mpc_hpipm.h        # HpipmInterface: HPIPM memory management and solve wrapper
├── src/
│   ├── lon_mpc.cpp            # LonMPC::Init/Update/Reset orchestration
│   ├── lon_mpc_model.cpp      # ModelBuilder implementation
│   └── lon_mpc_hpipm.cpp      # HpipmInterface implementation
└── test/
    ├── CMakeLists.txt
    ├── test_lon_mpc.cpp       # Integration test: constant velocity tracking, step response
    └── plot_result.py         # Visualization script for test output
```

```
thirdparty/
├── blasfeo/                   # BLASFEO source (git submodule)
│   └── CMakeLists.txt         # (upstream)
└── hpipm/                     # HPIPM source (git submodule)
    └── CMakeLists.txt         # (upstream)
```

---

### Task 1: Add HPIPM and BLASFEO as thirdparty dependencies

**Files:**
- Create: `thirdparty/blasfeo/` (git submodule)
- Create: `thirdparty/hpipm/` (git submodule)
- Modify: `CMakeLists.txt` (root, add subdirectories)

- [ ] **Step 1: Add BLASFEO as git submodule**

```bash
cd /home/lihongl/Desktop/mmt_ctrl/mmtctrl/control-dev_eu
git submodule add https://github.com/giaf/blasfeo.git thirdparty/blasfeo
cd thirdparty/blasfeo
git checkout 0.1.3
```

- [ ] **Step 2: Add HPIPM as git submodule**

```bash
cd /home/lihongl/Desktop/mmt_ctrl/mmtctrl/control-dev_eu
git submodule add https://github.com/giaf/hpipm.git thirdparty/hpipm
cd thirdparty/hpipm
git checkout 0.1.3
```

- [ ] **Step 3: Configure BLASFEO build options**

Create `thirdparty/blasfeo/CMakeLists_wrapper.cmake` is NOT needed — BLASFEO's own CMakeLists.txt works directly. But we need to set options before `add_subdirectory`. Add to root `CMakeLists.txt` before the thirdparty section:

```cmake
# BLASFEO and HPIPM — disable -Werror for thirdparty C code
set(CMAKE_C_FLAGS_SAVED "${CMAKE_C_FLAGS}")
string(REPLACE "-Werror" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

set(BLASFEO_HEADERS_INSTALLATION_DIRECTORY "${CMAKE_BINARY_DIR}/blasfeo_headers" CACHE STRING "" FORCE)
set(TARGET "GENERIC" CACHE STRING "" FORCE)
set(BLASFEO_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/blasfeo)

set(HPIPM_HEADERS_INSTALLATION_DIRECTORY "${CMAKE_BINARY_DIR}/hpipm_headers" CACHE STRING "" FORCE)
set(HPIPM_BLASFEO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/blasfeo" CACHE STRING "" FORCE)
set(HPIPM_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/hpipm)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVED}")
```

Insert these lines in root `CMakeLists.txt` after line 74 (`add_subdirectory(thirdparty/ilqr_lib)`) and before line 75 (`add_subdirectory(src/library/ctrl_common_lib)`).

- [ ] **Step 4: Verify build compiles**

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make blasfeo hpipm -j$(nproc)
```

Expected: Both libraries compile without error.

- [ ] **Step 5: Commit**

```bash
git add thirdparty/blasfeo thirdparty/hpipm CMakeLists.txt .gitmodules
git commit -m "feat: add BLASFEO and HPIPM as thirdparty dependencies"
```

---

### Task 2: Create lon_mpc_lib skeleton with public API headers

**Files:**
- Create: `src/library/lon_mpc_lib/CMakeLists.txt`
- Create: `src/library/lon_mpc_lib/include/lon_mpc.h`
- Modify: `CMakeLists.txt` (root, add subdirectory)

- [ ] **Step 1: Create CMakeLists.txt for lon_mpc_lib**

Create `src/library/lon_mpc_lib/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(lon_mpc_lib)

file(GLOB_RECURSE CPP_FILES src/*.cpp)

add_library(${PROJECT_NAME} STATIC ${CPP_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE hpipm blasfeo)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_BINARY_DIR}/blasfeo_headers
  ${CMAKE_BINARY_DIR}/hpipm_headers
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

add_subdirectory(test)
```

- [ ] **Step 2: Create public API header**

Create `src/library/lon_mpc_lib/include/lon_mpc.h`:

```cpp
#pragma once
#ifndef LON_MPC_H
#define LON_MPC_H

#include <memory>
#include <vector>

namespace lon_mpc {

struct LonMPCConfig {
  int N = 20;
  double dt = 0.1;

  // state tracking weights
  double q_s = 3.0;
  double q_v = 30.0;
  double q_a = 1.0;

  // terminal weights
  double q_s_N = 3.0;
  double q_v_N = 30.0;
  double q_a_N = 1.0;

  // control weight
  double q_jerk = 1.0;

  // soft constraint penalty
  double rho_v = 100.0;
  double rho_a = 100.0;

  // hard constraint: jerk
  double j_min = -5.0;
  double j_max = 3.0;

  // soft constraint: velocity
  double v_min = 0.0;
  double v_max = 60.0;

  // soft constraint: acceleration
  double a_min = -4.0;
  double a_max = 2.0;

  // fallback
  int max_fail_count = 3;
};

struct LonMPCInput {
  double s0 = 0.0;
  double v0 = 0.0;
  double a0 = 0.0;

  std::vector<double> s_ref;
  std::vector<double> v_ref;
  std::vector<double> a_ref;

  // runtime constraint override (use config defaults if not set)
  double v_min = 0.0;
  double v_max = 0.0;
  double a_min = 0.0;
  double a_max = 0.0;
  double j_min = 0.0;
  double j_max = 0.0;
  bool use_runtime_constraints = false;
};

struct LonMPCOutput {
  double a_des = 0.0;
  double jerk_des = 0.0;

  std::vector<double> s_pred;
  std::vector<double> v_pred;
  std::vector<double> a_pred;
  std::vector<double> jerk_pred;

  int solver_status = -1;
  int iter_count = 0;
  double max_residual = 0.0;
  double solve_time_ms = 0.0;
};

class LonMPC {
public:
  LonMPC();
  ~LonMPC();
  LonMPC(const LonMPC&) = delete;
  LonMPC& operator=(const LonMPC&) = delete;

  void Init(const LonMPCConfig& config);
  LonMPCOutput Update(const LonMPCInput& input);
  void Reset();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace lon_mpc

#endif // LON_MPC_H
```

- [ ] **Step 3: Add lon_mpc_lib to root CMakeLists.txt**

Add after line 80 (`add_subdirectory(src/library/mpc_cp_lib)`):

```cmake
add_subdirectory(src/library/lon_mpc_lib)
```

- [ ] **Step 4: Create placeholder source and test files**

Create `src/library/lon_mpc_lib/src/lon_mpc.cpp`:

```cpp
#include "lon_mpc.h"
#include <memory>

namespace lon_mpc {

struct LonMPC::Impl {
  LonMPCConfig config;
  bool initialized = false;
};

LonMPC::LonMPC() : impl_(std::make_unique<Impl>()) {}
LonMPC::~LonMPC() = default;

void LonMPC::Init(const LonMPCConfig& config) {
  impl_->config = config;
  impl_->initialized = true;
}

LonMPCOutput LonMPC::Update(const LonMPCInput& input) {
  LonMPCOutput output;
  output.solver_status = -1;
  return output;
}

void LonMPC::Reset() {
  impl_->initialized = false;
}

} // namespace lon_mpc
```

Create `src/library/lon_mpc_lib/test/CMakeLists.txt`:

```cmake
project(lon_mpc_test)

add_executable(lon_mpc_test test_lon_mpc.cpp)
target_link_libraries(lon_mpc_test PRIVATE lon_mpc_lib)
target_compile_features(lon_mpc_test PRIVATE cxx_std_17)
```

Create `src/library/lon_mpc_lib/test/test_lon_mpc.cpp`:

```cpp
#include "lon_mpc.h"
#include <cassert>
#include <cstdio>

int main() {
  lon_mpc::LonMPC mpc;
  lon_mpc::LonMPCConfig config;
  mpc.Init(config);

  lon_mpc::LonMPCInput input;
  input.v0 = 10.0;
  input.s_ref.resize(config.N + 1, 0.0);
  input.v_ref.resize(config.N + 1, 10.0);
  input.a_ref.resize(config.N + 1, 0.0);
  for (int i = 0; i <= config.N; i++) {
    input.s_ref[i] = 10.0 * i * config.dt;
  }

  auto output = mpc.Update(input);
  printf("solver_status: %d\n", output.solver_status);
  printf("Skeleton test passed.\n");
  return 0;
}
```

- [ ] **Step 5: Verify build**

```bash
cd build
cmake ..
make lon_mpc_lib lon_mpc_test -j$(nproc)
./src/library/lon_mpc_lib/test/lon_mpc_test
```

Expected: prints "Skeleton test passed."

- [ ] **Step 6: Commit**

```bash
git add src/library/lon_mpc_lib/ CMakeLists.txt
git commit -m "feat: add lon_mpc_lib skeleton with public API"
```

---

### Task 3: Implement ModelBuilder

**Files:**
- Create: `src/library/lon_mpc_lib/include/lon_mpc_model.h`
- Create: `src/library/lon_mpc_lib/src/lon_mpc_model.cpp`

- [ ] **Step 1: Create ModelBuilder header**

Create `src/library/lon_mpc_lib/include/lon_mpc_model.h`:

```cpp
#pragma once
#ifndef LON_MPC_MODEL_H
#define LON_MPC_MODEL_H

#include "lon_mpc.h"
#include <array>
#include <vector>

namespace lon_mpc {

constexpr int NX = 3;
constexpr int NU = 1;

struct StageData {
  // dynamics: x(k+1) = A*x(k) + B*u(k)
  double A[NX * NX] = {};
  double B[NX * NU] = {};
  double b[NX] = {};

  // cost: 0.5*x'Qx + q'x + 0.5*u'Ru + r'u
  double Q[NX * NX] = {};
  double q[NX] = {};
  double R[NU * NU] = {};
  double r[NU] = {};

  // state box constraint (soft for v, a)
  double lbx[NX] = {};
  double ubx[NX] = {};
  // which state indices are soft
  int soft_idx[2] = {1, 2};
  int n_soft = 2;
  // soft penalty (quadratic): Zl, Zu
  double Zl[2] = {};
  double Zu[2] = {};
  // soft penalty (linear): zl, zu
  double zl[2] = {};
  double zu[2] = {};

  // control box constraint (hard)
  double lbu[NU] = {};
  double ubu[NU] = {};
};

class ModelBuilder {
public:
  void Init(const LonMPCConfig& config);
  void UpdateConstraints(const LonMPCConfig& config);
  void BuildStages(const LonMPCInput& input);

  const StageData& GetStage(int k) const { return stages_[k]; }
  const StageData& GetTerminal() const { return terminal_; }
  int GetN() const { return N_; }

private:
  int N_ = 20;
  double dt_ = 0.1;
  LonMPCConfig config_;

  // discrete model matrices (computed once)
  double Ad_[NX * NX] = {};
  double Bd_[NX * NU] = {};

  std::vector<StageData> stages_;
  StageData terminal_;

  void ComputeDiscreteModel();
  void FillDynamics(StageData& stage);
  void FillCost(StageData& stage, const double* x_ref);
  void FillTerminalCost(StageData& stage, const double* x_ref);
  void FillConstraints(StageData& stage, bool is_terminal);
};

} // namespace lon_mpc

#endif // LON_MPC_MODEL_H
```

- [ ] **Step 2: Implement ModelBuilder**

Create `src/library/lon_mpc_lib/src/lon_mpc_model.cpp`:

```cpp
#include "lon_mpc_model.h"
#include <cstring>

namespace lon_mpc {

void ModelBuilder::Init(const LonMPCConfig& config) {
  config_ = config;
  N_ = config.N;
  dt_ = config.dt;
  stages_.resize(N_);
  ComputeDiscreteModel();
}

void ModelBuilder::UpdateConstraints(const LonMPCConfig& config) {
  config_.v_min = config.v_min;
  config_.v_max = config.v_max;
  config_.a_min = config.a_min;
  config_.a_max = config.a_max;
  config_.j_min = config.j_min;
  config_.j_max = config.j_max;
}

void ModelBuilder::ComputeDiscreteModel() {
  double dt = dt_;
  double dt2 = dt * dt;
  double dt3 = dt * dt * dt;

  // Ad = [1   dt   0.5*dt^2]
  //      [0   1    dt      ]
  //      [0   0    1       ]
  // stored column-major for HPIPM
  std::memset(Ad_, 0, sizeof(Ad_));
  Ad_[0 + 0 * NX] = 1.0;
  Ad_[1 + 0 * NX] = 0.0;
  Ad_[2 + 0 * NX] = 0.0;
  Ad_[0 + 1 * NX] = dt;
  Ad_[1 + 1 * NX] = 1.0;
  Ad_[2 + 1 * NX] = 0.0;
  Ad_[0 + 2 * NX] = 0.5 * dt2;
  Ad_[1 + 2 * NX] = dt;
  Ad_[2 + 2 * NX] = 1.0;

  // Bd = [(1/6)*dt^3]
  //      [0.5*dt^2  ]
  //      [dt        ]
  Bd_[0] = dt3 / 6.0;
  Bd_[1] = 0.5 * dt2;
  Bd_[2] = dt;
}

void ModelBuilder::BuildStages(const LonMPCInput& input) {
  for (int k = 0; k < N_; k++) {
    double x_ref[NX] = {input.s_ref[k], input.v_ref[k], input.a_ref[k]};
    FillDynamics(stages_[k]);
    FillCost(stages_[k], x_ref);
    FillConstraints(stages_[k], false);
  }
  double x_ref_N[NX] = {input.s_ref[N_], input.v_ref[N_], input.a_ref[N_]};
  FillTerminalCost(terminal_, x_ref_N);
  FillConstraints(terminal_, true);
}

void ModelBuilder::FillDynamics(StageData& stage) {
  std::memcpy(stage.A, Ad_, sizeof(Ad_));
  std::memcpy(stage.B, Bd_, sizeof(Bd_));
  std::memset(stage.b, 0, sizeof(stage.b));
}

void ModelBuilder::FillCost(StageData& stage, const double* x_ref) {
  // Q = diag(q_s, q_v, q_a), stored column-major
  std::memset(stage.Q, 0, sizeof(stage.Q));
  stage.Q[0 + 0 * NX] = config_.q_s;
  stage.Q[1 + 1 * NX] = config_.q_v;
  stage.Q[2 + 2 * NX] = config_.q_a;

  // q = -Q * x_ref
  stage.q[0] = -config_.q_s * x_ref[0];
  stage.q[1] = -config_.q_v * x_ref[1];
  stage.q[2] = -config_.q_a * x_ref[2];

  // R = [q_jerk]
  stage.R[0] = config_.q_jerk;
  stage.r[0] = 0.0;
}

void ModelBuilder::FillTerminalCost(StageData& stage, const double* x_ref) {
  std::memset(stage.Q, 0, sizeof(stage.Q));
  stage.Q[0 + 0 * NX] = config_.q_s_N;
  stage.Q[1 + 1 * NX] = config_.q_v_N;
  stage.Q[2 + 2 * NX] = config_.q_a_N;

  stage.q[0] = -config_.q_s_N * x_ref[0];
  stage.q[1] = -config_.q_v_N * x_ref[1];
  stage.q[2] = -config_.q_a_N * x_ref[2];

  // no control at terminal stage
  stage.R[0] = 0.0;
  stage.r[0] = 0.0;
}

void ModelBuilder::FillConstraints(StageData& stage, bool is_terminal) {
  // state bounds (s is unconstrained: use large values)
  stage.lbx[0] = -1e8;
  stage.ubx[0] = 1e8;
  stage.lbx[1] = config_.v_min;
  stage.ubx[1] = config_.v_max;
  stage.lbx[2] = config_.a_min;
  stage.ubx[2] = config_.a_max;

  // soft constraint indices and penalties
  stage.soft_idx[0] = 1;
  stage.soft_idx[1] = 2;
  stage.n_soft = 2;
  stage.Zl[0] = config_.rho_v;
  stage.Zu[0] = config_.rho_v;
  stage.Zl[1] = config_.rho_a;
  stage.Zu[1] = config_.rho_a;
  stage.zl[0] = 0.0;
  stage.zu[0] = 0.0;
  stage.zl[1] = 0.0;
  stage.zu[1] = 0.0;

  // control bounds (only for non-terminal)
  if (!is_terminal) {
    stage.lbu[0] = config_.j_min;
    stage.ubu[0] = config_.j_max;
  }
}

} // namespace lon_mpc
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make lon_mpc_lib -j$(nproc)
```

Expected: compiles without error.

- [ ] **Step 4: Commit**

```bash
git add src/library/lon_mpc_lib/include/lon_mpc_model.h src/library/lon_mpc_lib/src/lon_mpc_model.cpp
git commit -m "feat: implement ModelBuilder for triple-integrator MPC"
```

---

### Task 4: Implement HpipmInterface

**Files:**
- Create: `src/library/lon_mpc_lib/include/lon_mpc_hpipm.h`
- Create: `src/library/lon_mpc_lib/src/lon_mpc_hpipm.cpp`

- [ ] **Step 1: Create HpipmInterface header**

Create `src/library/lon_mpc_lib/include/lon_mpc_hpipm.h`:

```cpp
#pragma once
#ifndef LON_MPC_HPIPM_H
#define LON_MPC_HPIPM_H

#include "lon_mpc_model.h"
#include <vector>

namespace lon_mpc {

struct HpipmResult {
  int status = -1;
  int iter_count = 0;
  double max_residual = 0.0;

  // x[k] for k=0..N, each is NX doubles
  std::vector<double> x_sol;
  // u[k] for k=0..N-1, each is NU doubles
  std::vector<double> u_sol;
};

class HpipmInterface {
public:
  HpipmInterface();
  ~HpipmInterface();
  HpipmInterface(const HpipmInterface&) = delete;
  HpipmInterface& operator=(const HpipmInterface&) = delete;

  void Init(int N, int nx, int nu, int ns);
  HpipmResult Solve(const ModelBuilder& model, const double* x0);
  void Reset();

private:
  struct HpipmData;
  HpipmData* data_ = nullptr;

  int N_ = 0;
  int nx_ = 0;
  int nu_ = 0;
  int ns_ = 0;
  bool initialized_ = false;
};

} // namespace lon_mpc

#endif // LON_MPC_HPIPM_H
```

- [ ] **Step 2: Implement HpipmInterface**

Create `src/library/lon_mpc_lib/src/lon_mpc_hpipm.cpp`:

```cpp
#include "lon_mpc_hpipm.h"

extern "C" {
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_timing.h>
}

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <memory>

namespace lon_mpc {

struct HpipmInterface::HpipmData {
  // dimension
  hpipm_size_t dim_size = 0;
  void* dim_mem = nullptr;
  struct d_ocp_qp_dim dim;

  // qp
  hpipm_size_t qp_size = 0;
  void* qp_mem = nullptr;
  struct d_ocp_qp qp;

  // solution
  hpipm_size_t sol_size = 0;
  void* sol_mem = nullptr;
  struct d_ocp_qp_sol sol;

  // ipm arg
  hpipm_size_t arg_size = 0;
  void* arg_mem = nullptr;
  struct d_ocp_qp_ipm_arg arg;

  // ipm workspace
  hpipm_size_t ws_size = 0;
  void* ws_mem = nullptr;
  struct d_ocp_qp_ipm_ws ws;
};

HpipmInterface::HpipmInterface() : data_(new HpipmData()) {}

HpipmInterface::~HpipmInterface() {
  Reset();
  delete data_;
}

void HpipmInterface::Init(int N, int nx, int nu, int ns) {
  N_ = N;
  nx_ = nx;
  nu_ = nu;
  ns_ = ns;

  // arrays for dimensions per stage
  std::vector<int> nx_vec(N + 1, nx);
  std::vector<int> nu_vec(N + 1, nu);
  nu_vec[N] = 0; // no control at terminal
  std::vector<int> nbx_vec(N + 1, nx);  // box constraints on all states
  std::vector<int> nbu_vec(N + 1, nu);
  nbu_vec[N] = 0;
  std::vector<int> ng_vec(N + 1, 0);    // no general constraints
  std::vector<int> ns_vec(N + 1, ns);   // soft constraints
  ns_vec[0] = 0;                        // stage 0: no soft (initial state fixed)
  std::vector<int> nsbx_vec(N + 1, ns); // soft box on states
  nsbx_vec[0] = 0;                      // stage 0: no soft box
  std::vector<int> nsbu_vec(N + 1, 0);
  std::vector<int> nsg_vec(N + 1, 0);

  // --- dim ---
  data_->dim_size = d_ocp_qp_dim_memsize(N);
  data_->dim_size = (data_->dim_size + 63) / 64 * 64; // align to 64 bytes
  data_->dim_mem = aligned_alloc(64, data_->dim_size);
  d_ocp_qp_dim_create(N, &data_->dim, data_->dim_mem);
  d_ocp_qp_dim_set_all(
      nx_vec.data(), nu_vec.data(), nbx_vec.data(), nbu_vec.data(),
      ng_vec.data(), nsbx_vec.data(), nsbu_vec.data(), nsg_vec.data(),
      &data_->dim);

  // --- qp ---
  data_->qp_size = d_ocp_qp_memsize(&data_->dim);
  data_->qp_size = (data_->qp_size + 63) / 64 * 64;
  data_->qp_mem = aligned_alloc(64, data_->qp_size);
  d_ocp_qp_create(&data_->dim, &data_->qp, data_->qp_mem);

  // --- sol ---
  data_->sol_size = d_ocp_qp_sol_memsize(&data_->dim);
  data_->sol_size = (data_->sol_size + 63) / 64 * 64;
  data_->sol_mem = aligned_alloc(64, data_->sol_size);
  d_ocp_qp_sol_create(&data_->dim, &data_->sol, data_->sol_mem);

  // --- ipm arg ---
  data_->arg_size = d_ocp_qp_ipm_arg_memsize(&data_->dim);
  data_->arg_size = (data_->arg_size + 63) / 64 * 64;
  data_->arg_mem = aligned_alloc(64, data_->arg_size);
  d_ocp_qp_ipm_arg_create(&data_->dim, &data_->arg, data_->arg_mem);
  d_ocp_qp_ipm_arg_set_default(SPEED, &data_->arg);
  // override defaults
  double mu0 = 1e1;
  double tol = 1e-6;
  int max_iter = 50;
  int warm_start = 1;
  d_ocp_qp_ipm_arg_set_mu0(&mu0, &data_->arg);
  d_ocp_qp_ipm_arg_set_tol_stat(&tol, &data_->arg);
  d_ocp_qp_ipm_arg_set_tol_eq(&tol, &data_->arg);
  d_ocp_qp_ipm_arg_set_tol_ineq(&tol, &data_->arg);
  d_ocp_qp_ipm_arg_set_tol_comp(&tol, &data_->arg);
  d_ocp_qp_ipm_arg_set_iter_max(&max_iter, &data_->arg);
  d_ocp_qp_ipm_arg_set_warm_start(&warm_start, &data_->arg);

  // --- ipm workspace ---
  data_->ws_size = d_ocp_qp_ipm_ws_memsize(&data_->dim, &data_->arg);
  data_->ws_size = (data_->ws_size + 63) / 64 * 64;
  data_->ws_mem = aligned_alloc(64, data_->ws_size);
  d_ocp_qp_ipm_ws_create(&data_->dim, &data_->arg, &data_->ws, data_->ws_mem);

  initialized_ = true;
}

HpipmResult HpipmInterface::Solve(const ModelBuilder& model, const double* x0) {
  HpipmResult result;
  if (!initialized_) {
    result.status = -1;
    return result;
  }

  int N = model.GetN();

  // set initial state constraint via lbx/ubx at stage 0
  // (we fix x0 by setting lbx[0] = ubx[0] = x0)
  // But first set all stage data

  for (int k = 0; k < N; k++) {
    const auto& s = model.GetStage(k);

    d_ocp_qp_set_A(k, (double*)s.A, &data_->qp);
    d_ocp_qp_set_B(k, (double*)s.B, &data_->qp);
    d_ocp_qp_set_b(k, (double*)s.b, &data_->qp);

    d_ocp_qp_set_Q(k, (double*)s.Q, &data_->qp);
    d_ocp_qp_set_q(k, (double*)s.q, &data_->qp);
    d_ocp_qp_set_R(k, (double*)s.R, &data_->qp);
    d_ocp_qp_set_r(k, (double*)s.r, &data_->qp);

    // box constraints on u
    int idx_u[1] = {0};
    d_ocp_qp_set_idxbu(k, idx_u, &data_->qp);
    d_ocp_qp_set_lbu(k, (double*)s.lbu, &data_->qp);
    d_ocp_qp_set_ubu(k, (double*)s.ubu, &data_->qp);

    // box constraints on x
    int idx_x[NX] = {0, 1, 2};
    d_ocp_qp_set_idxbx(k, idx_x, &data_->qp);
    if (k == 0) {
      // fix initial state
      d_ocp_qp_set_lbx(k, (double*)x0, &data_->qp);
      d_ocp_qp_set_ubx(k, (double*)x0, &data_->qp);
    } else {
      d_ocp_qp_set_lbx(k, (double*)s.lbx, &data_->qp);
      d_ocp_qp_set_ubx(k, (double*)s.ubx, &data_->qp);
    }

    // soft constraints (only for k > 0, since k=0 is fixed)
    if (k > 0) {
      d_ocp_qp_set_idxs_rev(k, (int*)s.soft_idx, &data_->qp);
      d_ocp_qp_set_Zl(k, (double*)s.Zl, &data_->qp);
      d_ocp_qp_set_Zu(k, (double*)s.Zu, &data_->qp);
      d_ocp_qp_set_zl(k, (double*)s.zl, &data_->qp);
      d_ocp_qp_set_zu(k, (double*)s.zu, &data_->qp);
    }
  }

  // terminal stage
  {
    const auto& s = model.GetTerminal();
    int k = N;
    d_ocp_qp_set_Q(k, (double*)s.Q, &data_->qp);
    d_ocp_qp_set_q(k, (double*)s.q, &data_->qp);

    int idx_x[NX] = {0, 1, 2};
    d_ocp_qp_set_idxbx(k, idx_x, &data_->qp);
    d_ocp_qp_set_lbx(k, (double*)s.lbx, &data_->qp);
    d_ocp_qp_set_ubx(k, (double*)s.ubx, &data_->qp);

    d_ocp_qp_set_idxs_rev(k, (int*)s.soft_idx, &data_->qp);
    d_ocp_qp_set_Zl(k, (double*)s.Zl, &data_->qp);
    d_ocp_qp_set_Zu(k, (double*)s.Zu, &data_->qp);
    d_ocp_qp_set_zl(k, (double*)s.zl, &data_->qp);
    d_ocp_qp_set_zu(k, (double*)s.zu, &data_->qp);
  }

  // solve
  auto t_start = std::chrono::high_resolution_clock::now();
  d_ocp_qp_ipm_solve(&data_->qp, &data_->sol, &data_->arg, &data_->ws);
  auto t_end = std::chrono::high_resolution_clock::now();

  result.solve_time_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  d_ocp_qp_ipm_get_status(&data_->ws, &result.status);
  d_ocp_qp_ipm_get_iter(&data_->ws, &result.iter_count);
  d_ocp_qp_ipm_get_max_res_stat(&data_->ws, &result.max_residual);

  // extract solution
  result.x_sol.resize((N + 1) * nx_);
  result.u_sol.resize(N * nu_);

  for (int k = 0; k <= N; k++) {
    d_ocp_qp_sol_get_x(k, &data_->sol, result.x_sol.data() + k * nx_);
  }
  for (int k = 0; k < N; k++) {
    d_ocp_qp_sol_get_u(k, &data_->sol, result.u_sol.data() + k * nu_);
  }

  return result;
}

void HpipmInterface::Reset() {
  if (data_->dim_mem) { std::free(data_->dim_mem); data_->dim_mem = nullptr; }
  if (data_->qp_mem) { std::free(data_->qp_mem); data_->qp_mem = nullptr; }
  if (data_->sol_mem) { std::free(data_->sol_mem); data_->sol_mem = nullptr; }
  if (data_->arg_mem) { std::free(data_->arg_mem); data_->arg_mem = nullptr; }
  if (data_->ws_mem) { std::free(data_->ws_mem); data_->ws_mem = nullptr; }
  initialized_ = false;
}

} // namespace lon_mpc
```

- [ ] **Step 3: Verify build**

```bash
cd build && cmake .. && make lon_mpc_lib -j$(nproc)
```

Expected: compiles without error.

- [ ] **Step 4: Commit**

```bash
git add src/library/lon_mpc_lib/include/lon_mpc_hpipm.h src/library/lon_mpc_lib/src/lon_mpc_hpipm.cpp
git commit -m "feat: implement HpipmInterface for OCP QP solving"
```

---

### Task 5: Wire up LonMPC::Update with ModelBuilder and HpipmInterface

**Files:**
- Modify: `src/library/lon_mpc_lib/src/lon_mpc.cpp`

- [ ] **Step 1: Implement full LonMPC::Update**

Replace `src/library/lon_mpc_lib/src/lon_mpc.cpp`:

```cpp
#include "lon_mpc.h"
#include "lon_mpc_model.h"
#include "lon_mpc_hpipm.h"
#include <cmath>

namespace lon_mpc {

struct LonMPC::Impl {
  LonMPCConfig config;
  LonMPCConfig original_config;  // preserved for fallback to defaults
  ModelBuilder model_builder;
  HpipmInterface hpipm;
  bool initialized = false;

  // fallback state
  LonMPCOutput last_good_output;
  int fail_counter = 0;
};

LonMPC::LonMPC() : impl_(std::make_unique<Impl>()) {}
LonMPC::~LonMPC() = default;

void LonMPC::Init(const LonMPCConfig& config) {
  impl_->config = config;
  impl_->original_config = config;
  impl_->model_builder.Init(config);
  impl_->hpipm.Init(config.N, NX, NU, 2); // 2 soft constraints (v, a)
  impl_->initialized = true;
  impl_->fail_counter = 0;
  impl_->last_good_output = LonMPCOutput{};
}

LonMPCOutput LonMPC::Update(const LonMPCInput& input) {
  LonMPCOutput output;

  if (!impl_->initialized) {
    output.solver_status = -1;
    return output;
  }

  // input validation
  int N = impl_->config.N;
  if ((int)input.s_ref.size() != N + 1 ||
      (int)input.v_ref.size() != N + 1 ||
      (int)input.a_ref.size() != N + 1) {
    output.solver_status = -1;
    return output;
  }

  // apply runtime constraint override (use original config as base)
  LonMPCConfig active_config = impl_->original_config;
  if (input.use_runtime_constraints) {
    active_config.v_min = input.v_min;
    active_config.v_max = input.v_max;
    active_config.a_min = input.a_min;
    active_config.a_max = input.a_max;
    active_config.j_min = input.j_min;
    active_config.j_max = input.j_max;
  }
  impl_->config = active_config;
  impl_->model_builder.UpdateConstraints(active_config);

  // build QP stages
  impl_->model_builder.BuildStages(input);

  // initial state
  double x0[NX] = {input.s0, input.v0, input.a0};

  // solve
  auto result = impl_->hpipm.Solve(impl_->model_builder, x0);

  // fill output
  output.solver_status = result.status;
  output.iter_count = result.iter_count;
  output.max_residual = result.max_residual;
  output.solve_time_ms = result.solve_time_ms;

  if (result.status == 0) {
    // success
    output.s_pred.resize(N + 1);
    output.v_pred.resize(N + 1);
    output.a_pred.resize(N + 1);
    output.jerk_pred.resize(N);

    for (int k = 0; k <= N; k++) {
      output.s_pred[k] = result.x_sol[k * NX + 0];
      output.v_pred[k] = result.x_sol[k * NX + 1];
      output.a_pred[k] = result.x_sol[k * NX + 2];
    }
    for (int k = 0; k < N; k++) {
      output.jerk_pred[k] = result.u_sol[k * NU];
    }

    // a_des = predicted acceleration at t+dt (next step)
    output.a_des = output.a_pred[1];
    output.jerk_des = output.jerk_pred[0];

    // NaN/Inf protection
    if (!std::isfinite(output.a_des)) {
      output.a_des = input.a0;
      output.jerk_des = 0.0;
      output.solver_status = -2;
    } else {
      impl_->last_good_output = output;
      impl_->fail_counter = 0;
    }
  } else {
    // solver failed: fallback to last good output (hold-last-acceleration)
    impl_->fail_counter++;
    output.a_des = impl_->last_good_output.a_des;
    output.jerk_des = 0.0;
    if (impl_->fail_counter > impl_->config.max_fail_count) {
      output.solver_status = -3;  // degraded: caller should switch to PID
    }
  }

  return output;
}

void LonMPC::Reset() {
  impl_->hpipm.Reset();
  impl_->hpipm.Init(impl_->original_config.N, NX, NU, 2);
  impl_->fail_counter = 0;
  impl_->last_good_output = LonMPCOutput{};
}

} // namespace lon_mpc
```

- [ ] **Step 2: Verify build**

```bash
cd build && cmake .. && make lon_mpc_lib lon_mpc_test -j$(nproc)
```

Expected: compiles without error.

- [ ] **Step 3: Commit**

```bash
git add src/library/lon_mpc_lib/src/lon_mpc.cpp
git commit -m "feat: wire LonMPC Update with ModelBuilder and HpipmInterface"
```

---

### Task 6: Write integration tests

**Files:**
- Modify: `src/library/lon_mpc_lib/test/test_lon_mpc.cpp`
- Create: `src/library/lon_mpc_lib/test/plot_result.py`

- [ ] **Step 1: Implement integration test**

Replace `src/library/lon_mpc_lib/test/test_lon_mpc.cpp`:

```cpp
#include "lon_mpc.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>

void test_constant_velocity() {
  printf("=== Test: constant velocity tracking ===\n");

  lon_mpc::LonMPC mpc;
  lon_mpc::LonMPCConfig config;
  config.N = 20;
  config.dt = 0.1;
  config.q_s = 1.0;
  config.q_v = 50.0;
  config.q_a = 1.0;
  config.q_s_N = 1.0;
  config.q_v_N = 50.0;
  config.q_a_N = 1.0;
  config.q_jerk = 0.1;
  mpc.Init(config);

  double target_v = 10.0;
  lon_mpc::LonMPCInput input;
  input.s0 = 0.0;
  input.v0 = 10.0;
  input.a0 = 0.0;

  input.s_ref.resize(config.N + 1);
  input.v_ref.resize(config.N + 1);
  input.a_ref.resize(config.N + 1);
  for (int i = 0; i <= config.N; i++) {
    input.s_ref[i] = target_v * i * config.dt;
    input.v_ref[i] = target_v;
    input.a_ref[i] = 0.0;
  }

  auto output = mpc.Update(input);

  printf("  solver_status: %d\n", output.solver_status);
  printf("  iter_count: %d\n", output.iter_count);
  printf("  a_des: %.6f\n", output.a_des);
  printf("  jerk_des: %.6f\n", output.jerk_des);
  printf("  solve_time_ms: %.3f\n", output.solve_time_ms);

  assert(output.solver_status == 0);
  assert(std::fabs(output.a_des) < 0.1);
  assert(std::fabs(output.jerk_des) < 0.1);
  printf("  PASSED\n\n");
}

void test_acceleration_step() {
  printf("=== Test: acceleration step (0 -> 2 m/s^2) ===\n");

  lon_mpc::LonMPC mpc;
  lon_mpc::LonMPCConfig config;
  config.N = 20;
  config.dt = 0.1;
  config.q_s = 0.1;
  config.q_v = 10.0;
  config.q_a = 5.0;
  config.q_s_N = 0.1;
  config.q_v_N = 10.0;
  config.q_a_N = 5.0;
  config.q_jerk = 1.0;
  mpc.Init(config);

  double v0 = 5.0;
  double a_target = 2.0;
  lon_mpc::LonMPCInput input;
  input.s0 = 0.0;
  input.v0 = v0;
  input.a0 = 0.0;

  input.s_ref.resize(config.N + 1);
  input.v_ref.resize(config.N + 1);
  input.a_ref.resize(config.N + 1);

  double s = 0.0, v = v0;
  for (int i = 0; i <= config.N; i++) {
    input.s_ref[i] = s;
    input.v_ref[i] = v;
    input.a_ref[i] = a_target;
    s += v * config.dt + 0.5 * a_target * config.dt * config.dt;
    v += a_target * config.dt;
  }

  auto output = mpc.Update(input);

  printf("  solver_status: %d\n", output.solver_status);
  printf("  a_des: %.6f (target: %.1f)\n", output.a_des, a_target);
  printf("  jerk_des: %.6f\n", output.jerk_des);
  printf("  solve_time_ms: %.3f\n", output.solve_time_ms);

  assert(output.solver_status == 0);
  assert(output.jerk_des > 0.0); // should be positive jerk to reach a=2
  printf("  PASSED\n\n");
}

void test_closed_loop_simulation() {
  printf("=== Test: closed-loop simulation (speed up 5->15 m/s) ===\n");

  lon_mpc::LonMPC mpc;
  lon_mpc::LonMPCConfig config;
  config.N = 20;
  config.dt = 0.1;
  config.q_s = 1.0;
  config.q_v = 50.0;
  config.q_a = 1.0;
  config.q_s_N = 1.0;
  config.q_v_N = 50.0;
  config.q_a_N = 1.0;
  config.q_jerk = 0.5;
  config.a_min = -3.0;
  config.a_max = 2.0;
  config.j_min = -5.0;
  config.j_max = 5.0;
  mpc.Init(config);

  double target_v = 15.0;
  double s = 0.0, v = 5.0, a = 0.0;
  int sim_steps = 100;

  std::ofstream csv("lon_mpc_sim.csv");
  csv << "step,s,v,a,a_des,jerk_des,status\n";

  for (int step = 0; step < sim_steps; step++) {
    lon_mpc::LonMPCInput input;
    input.s0 = s;
    input.v0 = v;
    input.a0 = a;

    input.s_ref.resize(config.N + 1);
    input.v_ref.resize(config.N + 1);
    input.a_ref.resize(config.N + 1);

    double s_pred = s, v_pred = v;
    for (int i = 0; i <= config.N; i++) {
      input.v_ref[i] = target_v;
      input.a_ref[i] = 0.0;
      input.s_ref[i] = s_pred;
      s_pred += target_v * config.dt;
    }

    auto output = mpc.Update(input);

    csv << step << "," << s << "," << v << "," << a << ","
        << output.a_des << "," << output.jerk_des << ","
        << output.solver_status << "\n";

    // apply control using same discrete model as MPC (triple integrator)
    double jerk = output.jerk_des;
    double dt = config.dt;
    s += v * dt + 0.5 * a * dt * dt + (1.0/6.0) * jerk * dt * dt * dt;
    v += a * dt + 0.5 * jerk * dt * dt;
    a += jerk * dt;
  }

  csv.close();
  printf("  Simulation complete. Final v=%.3f (target=%.1f)\n", v, target_v);
  assert(std::fabs(v - target_v) < 1.0);
  printf("  PASSED\n\n");
}

int main() {
  test_constant_velocity();
  test_acceleration_step();
  test_closed_loop_simulation();
  printf("All tests passed.\n");
  return 0;
}
```

- [ ] **Step 2: Create visualization script**

Create `src/library/lon_mpc_lib/test/plot_result.py`:

```python
import pandas as pd
import matplotlib.pyplot as plt
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else "lon_mpc_sim.csv"
df = pd.read_csv(csv_file)

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(df["step"], df["v"], label="v actual")
axes[0].set_ylabel("velocity (m/s)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df["step"], df["a"], label="a actual")
axes[1].plot(df["step"], df["a_des"], "--", label="a_des")
axes[1].set_ylabel("acceleration (m/s^2)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(df["step"], df["jerk_des"], label="jerk_des")
axes[2].set_ylabel("jerk (m/s^3)")
axes[2].legend()
axes[2].grid(True)

axes[3].plot(df["step"], df["s"], label="s")
axes[3].set_ylabel("position (m)")
axes[3].set_xlabel("step")
axes[3].legend()
axes[3].grid(True)

plt.suptitle("LonMPC Closed-Loop Simulation")
plt.tight_layout()
plt.savefig("lon_mpc_sim.png", dpi=100)
plt.show()
```

- [ ] **Step 3: Build and run tests**

```bash
cd build && cmake .. && make lon_mpc_test -j$(nproc)
./src/library/lon_mpc_lib/test/lon_mpc_test
```

Expected output:
```
=== Test: constant velocity tracking ===
  solver_status: 0
  ...
  PASSED

=== Test: acceleration step (0 -> 2 m/s^2) ===
  solver_status: 0
  ...
  PASSED

=== Test: closed-loop simulation (speed up 5->15 m/s) ===
  ...
  PASSED

All tests passed.
```

- [ ] **Step 4: (Optional) Visualize**

```bash
cd build
python3 ../src/library/lon_mpc_lib/test/plot_result.py lon_mpc_sim.csv
```

- [ ] **Step 5: Commit**

```bash
git add src/library/lon_mpc_lib/test/
git commit -m "feat: add integration tests and visualization for LonMPC"
```

---

### Task 7: Runtime constraint override support

**Files:**
- Modify: `src/library/lon_mpc_lib/src/lon_mpc_model.cpp`

- [ ] **Step 1: Update FillConstraints to use runtime values**

In `src/library/lon_mpc_lib/src/lon_mpc_model.cpp`, the `BuildStages` method already rebuilds with the current `config_` values. When `LonMPC::Update` detects `use_runtime_constraints == true`, it re-inits the model builder with updated config. This is already implemented in Task 5.

Verify by adding a test to `test_lon_mpc.cpp`:

```cpp
void test_runtime_constraints() {
  printf("=== Test: runtime constraint override ===\n");

  lon_mpc::LonMPC mpc;
  lon_mpc::LonMPCConfig config;
  config.a_max = 2.0;
  mpc.Init(config);

  lon_mpc::LonMPCInput input;
  input.s0 = 0.0;
  input.v0 = 5.0;
  input.a0 = 0.0;
  input.s_ref.resize(config.N + 1);
  input.v_ref.resize(config.N + 1);
  input.a_ref.resize(config.N + 1);

  for (int i = 0; i <= config.N; i++) {
    input.v_ref[i] = 20.0; // large step -> will hit a_max
    input.a_ref[i] = 0.0;
    input.s_ref[i] = 20.0 * i * config.dt;
  }

  // override constraint: tighter a_max
  input.use_runtime_constraints = true;
  input.a_max = 1.0;
  input.a_min = -3.0;
  input.v_min = -0.1;
  input.v_max = 60.0;
  input.j_min = -5.0;
  input.j_max = 5.0;

  auto output = mpc.Update(input);

  printf("  solver_status: %d\n", output.solver_status);
  printf("  a_des: %.6f (should be <= 1.0 + small slack)\n", output.a_des);

  assert(output.solver_status == 0);
  assert(output.a_des < 1.5); // allow small slack violation
  printf("  PASSED\n\n");
}
```

Add `test_runtime_constraints();` call in `main()`.

- [ ] **Step 2: Build and run**

```bash
cd build && cmake .. && make lon_mpc_test -j$(nproc)
./src/library/lon_mpc_lib/test/lon_mpc_test
```

Expected: all tests pass including the new one.

- [ ] **Step 3: Commit**

```bash
git add src/library/lon_mpc_lib/test/test_lon_mpc.cpp
git commit -m "test: add runtime constraint override test"
```

---

### Task 8: Warm-start support

**Files:**
- Modify: `src/library/lon_mpc_lib/include/lon_mpc_hpipm.h`
- Modify: `src/library/lon_mpc_lib/src/lon_mpc_hpipm.cpp`

- [ ] **Step 1: Add warm-start initialization from previous solution**

HPIPM's `warm_start = 1` setting (already configured in Task 4) tells the solver to use whatever is currently in the `sol` structure as starting point. Since we reuse the same `sol` object across calls, the previous solution is automatically used as warm-start.

Verify by running the closed-loop test and checking that `iter_count` decreases after the first few steps. Add instrumentation:

In `test_lon_mpc.cpp`, modify `test_closed_loop_simulation` to print iter_count:

```cpp
    if (step < 5 || step % 20 == 0) {
      printf("  step %d: iter=%d, solve_time=%.2fms\n",
             step, output.iter_count, output.solve_time_ms);
    }
```

- [ ] **Step 2: Build and verify**

```bash
cd build && cmake .. && make lon_mpc_test -j$(nproc)
./src/library/lon_mpc_lib/test/lon_mpc_test
```

Expected: iter_count should generally decrease or remain low after initial steps (warm-start effect).

- [ ] **Step 3: Commit**

```bash
git add src/library/lon_mpc_lib/
git commit -m "feat: verify warm-start support in HPIPM solver"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|-----------|
| 1 | Add HPIPM/BLASFEO thirdparty | Build infrastructure |
| 2 | Create lib skeleton + API | `lon_mpc.h` public interface |
| 3 | Implement ModelBuilder | Triple-integrator discretization + QP stage assembly |
| 4 | Implement HpipmInterface | HPIPM C API wrapper |
| 5 | Wire up LonMPC::Update | Full solve pipeline |
| 6 | Integration tests | Constant velocity, step, closed-loop sim |
| 7 | Runtime constraint override | Dynamic constraint test |
| 8 | Warm-start verification | Performance confirmation |
