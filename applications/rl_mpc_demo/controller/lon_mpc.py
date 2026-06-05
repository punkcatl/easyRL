import casadi as ca
import numpy as np


class LonMPC:
    """Longitudinal MPC using triple integrator model.

    State: [s, v, a] (position, velocity, acceleration)
    Control: j (jerk)
    Output: a_des (desired acceleration)
    """

    def __init__(self, N: int, dt: float, Q_v: float, Q_a: float, R_j: float,
                 a_min: float, a_max: float, j_min: float, j_max: float,
                 v_min: float, v_max: float):
        self.N = N
        self.dt = dt

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(3, N + 1)  # states [s; v; a] over horizon
        U = opti.variable(1, N)      # control (jerk) over horizon
        p_v_ref = opti.parameter()   # reference velocity
        p_x0 = opti.parameter(3)     # initial state

        # Cost function
        cost = 0
        for k in range(N):
            cost += Q_v * (X[1, k] - p_v_ref) ** 2
            cost += Q_a * X[2, k] ** 2
            cost += R_j * U[0, k] ** 2
        # Terminal cost
        cost += Q_v * (X[1, N] - p_v_ref) ** 2

        opti.minimize(cost)

        # Dynamics: exact discretization of triple integrator
        for k in range(N):
            s_next = (X[0, k] + dt * X[1, k]
                      + 0.5 * dt**2 * X[2, k]
                      + (1.0 / 6.0) * dt**3 * U[0, k])
            v_next = X[1, k] + dt * X[2, k] + 0.5 * dt**2 * U[0, k]
            a_next = X[2, k] + dt * U[0, k]
            opti.subject_to(X[0, k + 1] == s_next)
            opti.subject_to(X[1, k + 1] == v_next)
            opti.subject_to(X[2, k + 1] == a_next)

        # Initial state constraint
        opti.subject_to(X[:, 0] == p_x0)

        # Box constraints
        opti.subject_to(opti.bounded(v_min, X[1, :], v_max))
        opti.subject_to(opti.bounded(a_min, X[2, :], a_max))
        opti.subject_to(opti.bounded(j_min, U[0, :], j_max))

        # Solver options
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._p_v_ref = p_v_ref
        self._p_x0 = p_x0
        self._sol = None

    def solve(self, s: float, v: float, a: float, v_ref: float) -> float:
        """Solve MPC and return desired acceleration.

        Args:
            s: current position
            v: current velocity
            a: current acceleration
            v_ref: reference velocity

        Returns:
            a_des: desired acceleration for next step
        """
        self._opti.set_value(self._p_x0, [s, v, a])
        self._opti.set_value(self._p_v_ref, v_ref)

        # Warm start: shift previous solution by one step for a better initial guess
        if self._sol is not None:
            import numpy as _np
            X_prev = self._sol.value(self._X)   # (3, N+1)
            U_prev = self._sol.value(self._U)   # (1, N)
            X_warm = _np.hstack([X_prev[:, 1:], X_prev[:, -1:]])
            U_warm = _np.hstack([U_prev[:, 1:], U_prev[:, -1:]])
            self._opti.set_initial(self._X, X_warm)
            self._opti.set_initial(self._U, U_warm)

        try:
            self._sol = self._opti.solve()
            a_des = float(self._sol.value(self._X[2, 1]))
        except RuntimeError:
            # Solver failed: return current acceleration as safe fallback
            a_des = a
            self._sol = None

        return a_des

    def reset(self):
        """Reset internal state for new episode."""
        self._sol = None
