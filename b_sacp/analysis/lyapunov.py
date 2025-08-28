import numpy as np

def wolf_lle(flow, x0, dt, steps, p, eps0=1e-6, renorm=50):
    """
    Wolf-style LLE by shadow trajectory separation & periodic renormalization.
    flow: function x -> x_next (discrete step of dt)
    """
    x = x0.copy()
    x_p = (x0 + eps0*np.random.randn(*x0.shape)).copy()
    s = 0.0; m = 0
    for k in range(steps):
        x  = flow(x, p, dt)
        x_p= flow(x_p, p, dt)
        d = x_p - x; dist = np.linalg.norm(d) + 1e-15
        if (k+1) % renorm == 0:
            s += np.log(dist/eps0)
            m += 1
            x_p = x + eps0 * d/dist
    return (s/(m*renorm*dt)) if m>0 else np.nan

def make_ib3d_flow(rhs, p):
    def f(x, p, dt):
        # RK4 step
        k1 = rhs(0.0, x, p)
        k2 = rhs(0.0, x + 0.5*dt*k1, p)
        k3 = rhs(0.0, x + 0.5*dt*k2, p)
        k4 = rhs(0.0, x + dt*k3, p)
        return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return f

