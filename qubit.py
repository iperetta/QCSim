import cmath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gates import QGATES

class Qubit:
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
    def __repr__(self):
        return str(self.state())
    def validate(self):
        # Spherical coordinates in Bloch sphere (unit sphere)
        if abs(self.theta.imag) < np.finfo(float).eps: self.theta = self.theta.real
        if abs(self.phi.imag) < np.finfo(float).eps: self.phi = self.phi.real
        # -- zenith  0 ⩽ 𝜃 ⩽ π
        self.theta %= 2*np.pi
        if self.theta > np.pi: # if self.theta == π
            self.theta = 2*np.pi - self.theta
            self.phi += np.pi
        # -- azimuth 0 ⩽ 𝜑 < 2π
        self.phi %= 2*np.pi
        # Cartesian coordinates
        self.x = np.sin(self.theta)*np.cos(self.phi)
        if abs(self.x) < np.finfo(float).eps: self.x = 0.
        self.y = np.sin(self.theta)*np.sin(self.phi)
        if abs(self.y) < np.finfo(float).eps: self.y = 0.
        self.z = np.cos(self.theta)
        if abs(self.z) < np.finfo(float).eps: self.z = 0.
        # Probability amplitudes
        self.a = np.cos(self.theta*0.5) # a \in R
        if abs(self.a) < np.finfo(float).eps: self.a = 0.
        self.b = np.sin(self.theta*0.5)*np.exp(1j*self.phi) # b \in C
        if abs(self.b) < np.finfo(float).eps: self.b = 0.
        elif abs(self.b.imag) < np.finfo(float).eps: self.b = self.b.real
    ### Setting a qubit ###
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
        if abs(a.imag) < np.finfo(float).eps: a = a.real # protect
        elif abs(a.imag) > 0: # ensure a \in R
            r1, phi1 = abs(a), cmath.phase(a)
            r2, phi2 = abs(b), cmath.phase(b)
            a = r1
            b = r2*np.exp(1j*(phi2-phi1))
        _tot = abs(a)**2 + abs(b)**2
        if _tot == 0: _tot = 1
        _a = np.sqrt(abs(a)**2/_tot)
        _b = np.sqrt(abs(b)**2/_tot)
        a = _a if a >= 0 else -1*_a
        b = 0 if abs(b) < np.finfo(float).eps else b*(_b/abs(b))
        if a == 0 and b == 0: a = 1 # protect
        self.theta = 2*np.arccos(a)
        self.phi = np.arctan2(b.imag, b.real) #!
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
    ### Functions ###
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
        return 0 if np.random.rand() < self.pb_0() else 1
    def simulate(self, times=100):
        measurements = np.zeros((times, ))
        for i in range(times):
            measurements[i] = self.measure()
        plt.figure()
        plt.hist(measurements, bins=[-0.1, 0.1, 0.9, 1.1], orientation='vertical', density=True)
        plt.grid('on')
        plt.show()
    ### GATES ###
    def X_gate(self, to_self=True):
        sigma_x = QGATES['X']
        if to_self:
            self.set_state(sigma_x @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_x @ self.state())
        return aux
    def Z_gate(self, to_self=True):
        sigma_z = QGATES['Z']
        if to_self:
            self.set_state(sigma_z @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_z @ self.state())
        return aux
    def Y_gate(self, to_self=True):
        sigma_y = QGATES['Y']
        if to_self:
            self.set_state(sigma_y @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_y @ self.state())
        return aux
    def ID_gate(self, to_self=True):
        sigma_id = QGATES['ID']
        if to_self:
            self.set_state(sigma_id @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_id @ self.state())
        return aux
    def H_gate(self, to_self=True):
        sigma_h = QGATES['H']
        if to_self:
            self.set_state(sigma_h @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_h @ self.state())
        return aux
    def Rz_phi_gate(self, phi, to_self=True):
        sigma_rz = QGATES['Rz_phi'](phi)
        if to_self:
            self.set_state(sigma_rz @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_rz @ self.state())
        return aux
    def S_gate(self, to_self=True):
        sigma_s = QGATES['S']
        if to_self:
            self.set_state(sigma_s @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_s @ self.state())
        return aux
    def S_cross_gate(self, to_self=True):
        sigma_s = QGATES['S+']
        if to_self:
            self.set_state(sigma_s @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_s @ self.state())
        return aux
    def T_gate(self, to_self=True):
        sigma_t = QGATES['T']
        if to_self:
            self.set_state(sigma_t @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_t @ self.state())
        return aux
    def T_cross_gate(self, to_self=True):
        sigma_t = QGATES['T+']
        if to_self:
            self.set_state(sigma_t @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_t @ self.state())
        return aux
    def Rx_phi_gate(self, phi, to_self=True):
        sigma_rx = QGATES['Rx_phi'](phi)
        if to_self:
            self.set_state(sigma_rx @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_rx @ self.state())
        return aux
    def Ry_phi_gate(self, phi, to_self=True):
        sigma_ry = QGATES['Ry_phi'](phi)
        if to_self:
            self.set_state(sigma_ry @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_ry @ self.state())
        return aux
    def sqNOT_gate(self, to_self=True):
        sigma_sn = QGATES['sqNOT']
        if to_self:
            self.set_state(sigma_sn @ self.state())
            return
        aux = Qubit()
        aux.set_state(sigma_sn @ self.state())
        return aux
    def RESET_gate(self, to_self=True):
        """Use of this operation may make your code non-portable."""
        if to_self:
            self.set_ket('0')
            return
        aux = Qubit()
        aux.set_ket('0')
        return aux



class MultiQubits:
    # KET_PHIp = np.array([[np.sqrt(2)/2], [np.sqrt(2)/2]])  |00> |11>
    # KET_PHIm = np.array([[np.sqrt(2)/2], [-np.sqrt(2)/2]]) |00> |11>
    # KET_PSIp = np.array([[np.sqrt(2)/2], [np.sqrt(2)/2]])  |01> |10>
    # KET_PSIm = np.array([[np.sqrt(2)/2], [-np.sqrt(2)/2]]) |01> |10>
    def __init__(self, nr_qubits):
        if nr_qubits < 2:
            raise Exception("Consider using class Qubit for this")
        self.nr_qubits = nr_qubits
        s = list([1.] if i == 0 else [0.] for i in range(2**nr_qubits))
        self._state = np.array(s)
    def __getitem__(self, index):
        return self._state[index, 0]
    def state(self):
        return self._state
    def bin(self, num):
        b = bin(num)[2:]
        lb = len(b)
        return "0"*(self.nr_qubits - lb) + b \
            if lb <= self.nr_qubits else b[-self.nr_qubits:]
    def probabilities(self):
        return np.abs(self.state())**2
    def table_prob(self):
        t = "st" + " "*(self.nr_qubits - 2) + "| Prob\n"
        t += "--" + "-"*self.nr_qubits + "------\n"
        for i in range(2**self.nr_qubits):
            t += f"{self.bin(i)}| {100*abs(self[i])**2:.1f}%\n"
        return t
    def set(self, s):
        self._state = s
    def init_from_qubits(self, *args):
        if len(args) == self.nr_qubits:
            qubits = list(Qubit() for _ in range(self.nr_qubits))
            for i, arg in enumerate(args):
                if type(arg) in [int, str]:
                    qubits[i].set_ket(arg)
                elif type(arg) in [tuple, list]:
                    qubits[i].set(*arg)
            s = qubits[0].state()
            for q in qubits[1:]:
                s = np.kron(s, q.state())
            self._state = s
        else:
            raise Exception(f"Expecting {self.nr_qubits} qubits, not {len(args)} as informed")
    def measure(self):
        prob = list(p[0] for p in self.probabilities().tolist())
        chosen = np.random.choice(2**self.nr_qubits, size=None, p=prob, replace=True)
        return self.bin(chosen)
    def histogram(self, outcomes, labels, plot=True, perc=False):
        h = dict()
        bins = []
        for l in labels:
            c = outcomes.count(l)
            h[l] = c
            bins.append(c)
        if perc:
            total = len(outcomes)
            for i, l in enumerate(labels):
                h[l] /= total
                bins[i] /= total
        if plot:
            plt.rcParams.update({'font.size': 14})
            fig, ax = plt.subplots()
            ax.barh(labels, bins, align='center')
            ax.invert_yaxis()
            ax.grid('on')
            plt.title("Probabilities for outcomes")
            plt.xlabel("probability")
            plt.ylabel("outcome")
            plt.show()
        return h
    def simulate(self, times=100, plot=True, perc=True):
        measurements = []
        for i in range(times):
            measurements.append(self.measure())
        h = self.histogram(measurements, 
            list(self.bin(s) for s in range(2**self.nr_qubits)), 
            plot, perc)
        return h
    def multi_gate(self, *args):
        """Args is sequence of tuples each of (qubit_idx, gate) or (qubit_idx, gate, phi)"""
        sigma_id = QGATES['ID']
        sigmas = list(sigma_id for _ in range(self.nr_qubits))
        for arg in args:
            qubit_idx = arg[0]
            gate = arg[1]
            phi = None if len(arg) == 2 else arg[2]
            if 'phi' in gate and phi is None:
                raise Exception(f"Gate {gate} needs 'phi' parameter")
            sigmas[qubit_idx] = QGATES[gate]
        qg = sigmas[0]
        for i in range(1, self.nr_qubits):
            qg = np.kron(qg, sigmas[i])
        return qg
    def apply_gate(self, qubit_idx, gate, phi=None, to_self=True):
        sigma_gate = self.multi_gate((qubit_idx, gate, phi))
        if to_self:
            self.set(sigma_gate @ self.state())
            return
        aux = MultiQubits()
        aux.set(sigma_gate @ self.state())
        return aux
    def apply_gates(self, *args, to_self=True):
        sigma_gate = self.multi_gate(*args)
        print(sigma_gate)
        if to_self:
            self.set(sigma_gate @ self.state())
            return
        aux = MultiQubits()
        aux.set(sigma_gate @ self.state())
        return aux


    


if __name__ == '__main__':
    # # q = Qubit()
    # # q.set(1,1)
    # # print(q.state())
    # # q.simulate(1000)
    # qs = MultiQubits(3)
    # qs.init_from_qubits(
    #     # |0❭     |1❭
    #     # (1,0),
    #     # (1,1),
    #     # (1,0),
    #     # (1,      1  ),
    #     # (1,      1  ),
    #     # (1,      1  )
    #     (.3-.3j, 0.8   ),
    #     (1,      2     ),
    #     (0.8,    .3-.3j)
    # )
    # print(qs.table_prob())
    # # qs.set(0,(1,1))
    # s = qs.state()
    # print(s.size)
    # print(s)

    # print(qs.bin(1024))
    # # print('      00     01     10     11')
    # # for i in range(2):
    # #     for j in range(2):
    # #         qs.set(i, (3,3))
    # #         s = qs.state()
    # #         print(i, 's', f"{s.tolist()}'")

    # # for _ in range(20):
    # #     print(qs.measure(), end=', ')
    # # print(" ")

    # qs.simulate(1000)


    qs = MultiQubits(3)
    qs.apply_gates((0, 'X'), (2, 'H'))
    print(qs.state())
    qs.simulate(10000)