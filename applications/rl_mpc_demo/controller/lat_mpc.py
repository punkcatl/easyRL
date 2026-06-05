import casadi as ca
import numpy as np


class LatMPC:
    """Lateral MPC using kinematic bicycle model.

    State: [x, y, psi] (position_x, position_y, heading)
    Control: delta (front wheel steering angle)
    """

    def __init__(self, N: int, dt: float, L: float,
                 Q_y: float, Q_psi: float, R_delta: float,
                 delta_min: float, delta_max: float, delta_dot_max: float):
        self.N = N
        self.dt = dt
        self.L = L

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(3, N + 1)  # [x, y, psi]
        U = opti.variable(1, N)      # delta (steering)

        # Parameters
        p_x0 = opti.parameter(3)         # initial state [x0, y0, psi0]
        p_v = opti.parameter()           # longitudinal velocity
        p_y_ref = opti.parameter()       # reference lateral position
        p_psi_ref = opti.parameter()     # reference heading
        p_delta_prev = opti.parameter()  # previous steering angle (for rate limit)

        # Cost
        cost = 0
        for k in range(N):
            cost += Q_y * (X[1, k] - p_y_ref) ** 2
            cost += Q_psi * (X[2, k] - p_psi_ref) ** 2
            cost += R_delta * U[0, k] ** 2
        # Terminal cost
        cost += Q_y * (X[1, N] - p_y_ref) ** 2
        cost += Q_psi * (X[2, N] - p_psi_ref) ** 2

        opti.minimize(cost)

        # Kinematic bicycle dynamics (nonlinear)
        for k in range(N):
            x_next = X[0, k] + dt * p_v * ca.cos(X[2, k])
            y_next = X[1, k] + dt * p_v * ca.sin(X[2, k])
            psi_next = X[2, k] + dt * p_v / L * ca.tan(U[0, k])
            opti.subject_to(X[0, k + 1] == x_next)
            opti.subject_to(X[1, k + 1] == y_next)
            opti.subject_to(X[2, k + 1] == psi_next)

        # Initial state
        opti.subject_to(X[:, 0] == p_x0)

        # Steering angle bounds
        opti.subject_to(opti.bounded(delta_min, U[0, :], delta_max))

        # Steering rate constraint
        opti.subject_to(opti.bounded(
            -delta_dot_max * dt, U[0, 0] - p_delta_prev, delta_dot_max * dt))
        for k in range(N - 1):
            opti.subject_to(opti.bounded(
                -delta_dot_max * dt, U[0, k + 1] - U[0, k], delta_dot_max * dt))

        # Solver
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.warm_start_init_point": "yes",
        }
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._p_x0 = p_x0
        self._p_v = p_v
        self._p_y_ref = p_y_ref
        self._p_psi_ref = p_psi_ref
        self._p_delta_prev = p_delta_prev
        self._sol = None
        self._delta_prev = 0.0

    def solve(self, x: float, y: float, psi: float, v: float,
              y_ref: float, psi_ref: float = 0.0) -> float:
        """Solve lateral MPC and return steering angle.

        Args:
            x: current x position
            y: current y position
            psi: current heading angle
            v: current longitudinal velocity
            y_ref: reference lateral position
            psi_ref: reference heading (default 0 = straight)

        Returns:
            delta: front wheel steering angle (rad)
        """
        # Avoid division by zero at very low speed
        v_safe = max(v, 1.0)

        self._opti.set_value(self._p_x0, [x, y, psi])
        self._opti.set_value(self._p_v, v_safe)
        self._opti.set_value(self._p_y_ref, y_ref)
        self._opti.set_value(self._p_psi_ref, psi_ref)
        self._opti.set_value(self._p_delta_prev, self._delta_prev)

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
            delta = float(self._sol.value(self._U[0, 0]))
        except RuntimeError:
            delta = self._delta_prev
            self._sol = None

        self._delta_prev = delta
        return delta

    def reset(self):
        """Reset internal state for new episode."""
        self._sol = None
        self._delta_prev = 0.0
