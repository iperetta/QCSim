import cmath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from fractions import Fraction

class Qubit:
    @staticmethod
    def plot(*args, **kwargs):
        def repel_from_center(x, y, z, m=0.1):
            return x + (-m if x < 0 else m), \
                y + (-m if y < 0 else m), \
                z + (-m if z < 0 else m)
        def bloch_sphere():
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            u = np.linspace(0, 2*np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = 1 * np.outer(np.cos(u), np.sin(v))
            y = 1 * np.outer(np.sin(u), np.sin(v))
            z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_wireframe(x, y, z, color='gray', linestyle=':')
            ax.plot3D([-1, 1], [0, 0], [0, 0], color='k', linestyle='--')
            ax.text(-1.1, 0, 0, '|$-$❭', 'x', horizontalalignment='right', fontweight='bold')
            ax.text(1.1, 0, 0, '|$+$❭', 'x', horizontalalignment='left', fontweight='bold')
            ax.plot3D([0, 0], [-1, 1], [0, 0], color='k', linestyle='--')
            ax.text(0, -1.1, 0, '|$-i$❭', 'y', horizontalalignment='right', fontweight='bold')
            ax.text(0, 1.1, 0, '|$i$❭', 'y', horizontalalignment='left', fontweight='bold')
            ax.plot3D([0, 0], [0, 0], [-1, 1], color='k', linestyle='--')
            ax.text(0, 0, -1.1, '|1❭', 'x', horizontalalignment='center', fontweight='bold')
            ax.text(0, 0, 1.1, '|0❭', 'x', horizontalalignment='center', fontweight='bold')
            limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
            ax.set_box_aspect(np.ptp(limits, axis = 1))
            ax._axis3don = False
            return ax
        if kwargs.get('title', False):
            title = kwargs['title']
        else:
            title = ''
        ax = bloch_sphere()
        for arg in args:
            label, color = '|ψ❭', 'r'
            if type(arg) == tuple:
                if len(arg) == 3: color = arg[2]
                label = '|' + arg[1] + '❭'
                arg = arg[0]
            ax.quiver(0, 0, 0, arg.x, arg.y, arg.z, color=color)
            ax.text(*repel_from_center(arg.x, arg.y, arg.z), label, 'x', \
                horizontalalignment='center', fontweight='bold', color=color)
        plt.title(title)
        plt.show()
    KET_0 = np.array([[1.], [0.]])
    KET_1 = np.array([[0.], [1.]])
    def __init__(self):
        self.a = None
        self.b = None
        self.x = None
        self.y = None
        self.z = None
        self.theta = 0.  # zenith  0 ⩽ 𝜃 ⩽ π
        self.phi   = 0.  # azimuth 0 ⩽ 𝜑 ⩽ 2π
        self.validate()
    def validate(self):
        # Spherical coordinates in Bloch sphere (unit sphere)
        if abs(self.theta.imag) < np.finfo(float).eps: self.theta = self.theta.real
        if abs(self.phi.imag) < np.finfo(float).eps: self.phi = self.phi.real
        # -- zenith  0 ⩽ 𝜃 ⩽ π
        self.theta %= 2*np.pi
        if self.theta > np.pi: # if self.theta == π
            self.theta = 2*np.pi - self.theta
        # -- azimuth 0 ⩽ 𝜑 ⩽ 2π
        self.phi %= 2*np.pi
        # Cartesian coordinates
        self.x = np.sin(self.theta)*np.cos(self.phi)
        if abs(self.x) < np.finfo(float).eps: self.x = 0.
        self.y = np.sin(self.theta)*np.sin(self.phi)
        if abs(self.y) < np.finfo(float).eps: self.y = 0.
        self.z = np.cos(self.theta)
        if abs(self.z) < np.finfo(float).eps: self.z = 0.
        # Probability amplitudes
        self.a = np.cos(self.theta*0.5)
        if abs(self.a) < np.finfo(float).eps: self.a = 0.
        self.b = np.sin(self.theta*0.5)*np.exp(1j*self.phi)
        if abs(self.b) < np.finfo(float).eps: self.b = 0.
        elif abs(self.b.imag) < np.finfo(float).eps: self.b = self.b.real
    def set_cartesian(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            self.theta = 0
            self.phi = 0
        else:
            self.theta = np.arccos(z/r)
            self.phi = np.arctan2(y, x)
        self.validate()
    def set_spherical(self, theta, phi):
        self.theta = theta
        self.phi = phi
        self.validate()
    def set_probability_amplitudes(self, a, b):
        # _tot = abs(a)**2 + abs(b)**2
        # _a = np.sqrt(abs(a)**2/_tot)
        # _b = np.sqrt(abs(b)**2/_tot)
        # a = _a if a > 0 else -1*_a
        # b = _b if b > 0 else -1*_b
        if abs(a.imag) < np.finfo(float).eps: a = a.real
        elif abs(a.imag) > 0: 
            r1, e1 = abs(a), cmath.phase(a)
            r2, e2 = abs(b), cmath.phase(b)
            a = r1
            b = r2*np.exp(1j*(e2-e1))
        _tot = abs(a)**2 + abs(b)**2
        if _tot == 0: _tot = 1
        _a = np.sqrt(abs(a)**2/_tot)
        _b = np.sqrt(abs(b)**2/_tot)
        a = _a if a >= 0 else -1*_a
        b = b*(_b/abs(b)) if abs(b) > 0 else 0
        if a == 0 and b == 0: a = 1
        self.theta = 2*np.arccos(a)
        self.phi = np.arctan2(b.imag, b.real)
        self.validate()
    def set_ket(self, symbol):
        if type(symbol) == int:
            symbol = str(symbol)
        if symbol == '0':
            self.set_cartesian(0, 0, 1)
        elif symbol == '1':
            self.set_cartesian(0, 0, -1)
        elif symbol == '+':
            self.set_cartesian(1, 0, 0)
        elif symbol == '-':
            self.set_cartesian(-1, 0, 0)
        elif symbol == 'i':
            self.set_cartesian(0, 1, 0)
        elif symbol == '-i':
            self.set_cartesian(0, -1, 0)
        else:
            raise Exception(f"symbol {symbol} unrecognized; accepted: 0, 1, +, -, i, -i")
    def set(self, a, b):
        self.set_probability_amplitudes(a, b)
    def set_state(self, array):
        a = array[0, 0]
        b = array[1, 0]
        self.set_probability_amplitudes(a, b)
    def set_as(self, other):
        self.set_probability_amplitudes(other.a, other.b)
    def ket(self):
        return self.a*Qubit.KET_0 + self.b*Qubit.KET_1
    def state(self):
        return np.array([[self.a], [self.b]])
    def vector(self):
        return np.array([[self.x], [self.y], [self.z]])
    def probability(self, ket):
        return abs(self.a)**2 if ket == 0 else abs(self.b)**2
    def pb_0(self):
        return self.probability(0)
    def pb_1(self):
        return self.probability(1)
    def measure(self):
        return [0] if np.random.rand() < self.pb_0() else [1]
    ### GATES ###
    def X_gate(self):
        aux = Qubit()
        sigma_x = np.array([[0, 1], [1, 0]])
        aux.set_state(sigma_x @ self.state())
        return aux
    # def X_gate_alt(self):
    #     aux = Qubit()
    #     sigma_x = np.array([
    #         [ 1,  0,  0], 
    #         [ 0, -1,  0],
    #         [ 0,  0, -1],
    #     ])
    #     tmp = sigma_x @ self.vector()
    #     aux.set_cartesian(tmp[0,0], tmp[1,0], tmp[2,0])
    #     return aux
    def Z_gate(self):
        aux = Qubit()
        sigma_z = np.array([[1, 0], [0, -1]])
        aux.set_state(sigma_z @ self.state())
        return aux
    def Y_gate(self):
        aux = Qubit()
        sigma_y = 1j*np.array([[0, -1], [1, 0]])
        aux.set_state(sigma_y @ self.state())
        return aux
    def ID_gate(self):
        aux = Qubit()
        sigma_id = np.array([[1, 0], [0, 1]])
        aux.set_state(sigma_id @ self.state())
        return aux
    def H_gate(self):
        aux = Qubit()
        sigma_h = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
        aux.set_state(sigma_h @ self.state())
        return aux
    def Rz_phi_gate(self, phi):
        aux = Qubit()
        sigma_rz = np.array([[1, 0], [0, np.exp(phi*1j)]])
        aux.set_state(sigma_rz @ self.state())
        return aux
    def S_gate(self):
        return self.Rz_phi_gate(np.pi/2)
    def S_cross_gate(self):
        return self.Rz_phi_gate(3*np.pi/2)
    def T_gate(self):
        return self.Rz_phi_gate(np.pi/4)
    def T_cross_gate(self):
        return self.Rz_phi_gate(7*np.pi/4)
    def Rx_phi_gate(self, phi):
        aux = Qubit()
        sigma_rx = np.array([
            [np.cos(phi/2), -np.sin(phi/2)*1j], 
            [-np.sin(phi/2)*1j, np.cos(phi/2)]
        ])
        aux.set_state(sigma_rx @ self.state())
        return aux
    def Ry_phi_gate(self, phi):
        aux = Qubit()
        sigma_ry = np.array([
            [np.cos(phi/2), -np.sin(phi/2)], 
            [np.sin(phi/2), np.cos(phi/2)]
        ])
        aux.set_state(sigma_ry @ self.state())
        return aux
    def sqNOT_gate(self):
        aux = Qubit()
        sigma_sn = 0.5*np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        aux.set_state(sigma_sn @ self.state())
        return aux
    def RESET_gate(self):
        """Use of this operation may make your code non-portable."""
        aux = Qubit()
        aux.set_ket('0')
        return aux
    ### ###
    def __repr__(self):
        return str(self.state())


if __name__ == '__main__':
    q = Qubit()
    print(q.state())

    q.set_probability_amplitudes(np.sqrt(2)/2, np.sqrt(2)/2)
    q.set_probability_amplitudes(0.5, 0.5)
    print(q.state())
    print(q.vector())
    # print(q.probability(1))
    print('Prob{|0>}:', q.pb_0())
    print('Prob{|1>}:', q.pb_1())
    q.set_cartesian(np.sqrt(2)/2, np.sqrt(2)/2, 0)
    print(q.state())
    print(q.vector())
    print('Prob{|0>}:', q.pb_0())
    print('Prob{|1>}:', q.pb_1())

    q.set_cartesian(5, 5, 0)
    print(q.state())
    print(q.vector())
    print('Prob{|0>}:', q.pb_0())
    print('Prob{|1>}:', q.pb_1())

    q.set_spherical(np.pi/2, np.pi/4)
    print(q.state())
    print(q.vector())
    print('Prob{|0>}:', q.pb_0())
    print('Prob{|1>}:', q.pb_1())

    for i in range(100):
        print(q.measure())

    q.set_cartesian(1, 1, 1)
    p = q.X_gate()
    print(p.state())
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="X gate") 

    # q.set_cartesian(1, 1, 1)
    p = q.Z_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="Z gate") 

    # q.set_cartesian(0, 1, 0)
    p = q.Y_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="Y gate") 

    p = q.ID_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="ID gate") 

    q.set_ket('-')
    p = q.H_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="H gate") 

    q.set_cartesian(1, 1, 1)
    phi = np.pi
    p = q.Rz_phi_gate(phi)
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title=f"$R^z_\phi$ gate, $\phi={phi:.3f}$ rad") 

    phi = np.pi
    p = q.Rx_phi_gate(phi)
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title=f"$R^x_\phi$ gate, $\phi={phi:.3f}$ rad") 

    phi = np.pi
    p = q.Ry_phi_gate(phi)
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title=f"$R^y_\phi$ gate, $\phi={phi:.3f}$ rad") 


    p = q.S_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="S gate") 

    p = q.S_cross_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="S† gate") 

    p = q.T_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="T gate") 

    p = q.T_cross_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="T† gate") 

    p = q.sqNOT_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="$\sqrt{NOT}$ gate") 

    p = q.RESET_gate()
    Qubit.plot((q, 'q'), (p, 'p', 'b'), title="RESET gate") 


    for i in list((n-18)/18 for n in range(37)):
        print(np.arccos(i))