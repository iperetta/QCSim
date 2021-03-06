import cmath
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.shape_base import tile

# import qugates as qgt
from qugate import QuGates as qg

TOL = 2*np.finfo(float).eps

class QuBit:
    @staticmethod
    def clean_matrix(a):
        m, n = a.shape
        for i in range(m):
            for j in range(n):
                if abs(a[i, j]) < TOL:
                    a[i, j] = 0.
                if abs(a[i, j].real) < TOL:
                    a[i, j] -= a[i, j].real # discard real part
                if abs(a[i, j].imag) < TOL:
                    a[i, j] = a[i, j].real  # keep real part only
        return a
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
        if abs(self.theta.imag) < TOL: self.theta = self.theta.real
        if abs(self.phi.imag) < TOL: self.phi = self.phi.real
        # -- zenith  0 ⩽ 𝜃 ⩽ π
        self.theta %= 2*np.pi
        if self.theta > np.pi: # if self.theta == π
            self.theta = 2*np.pi - self.theta
            self.phi += np.pi
        # -- azimuth 0 ⩽ 𝜑 < 2π
        self.phi %= 2*np.pi
        # Cartesian coordinates
        self.x = np.sin(self.theta)*np.cos(self.phi)
        if abs(self.x) < TOL: self.x = 0.
        self.y = np.sin(self.theta)*np.sin(self.phi)
        if abs(self.y) < TOL: self.y = 0.
        self.z = np.cos(self.theta)
        if abs(self.z) < TOL: self.z = 0.
        # Probability amplitudes
        self.a = np.cos(self.theta*0.5) # a \in R
        if abs(self.a) < TOL: self.a = 0.
        self.b = np.sin(self.theta*0.5)*np.exp(1j*self.phi) # b \in C
        if abs(self.b) < TOL: self.b = 0.
        elif abs(self.b.imag) < TOL: self.b = self.b.real
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
        if abs(a.imag) < TOL: a = a.real # protect
        elif abs(a.imag) > TOL: # ensure a \in R
            r1, phi1 = abs(a), cmath.phase(a)
            r2, phi2 = abs(b), cmath.phase(b)
            a = r1
            b = r2*np.exp(1j*(phi2-phi1))
        _tot = abs(a)**2 + abs(b)**2
        if _tot == 0: _tot = 1
        _a = np.sqrt(abs(a)**2/_tot)
        _b = np.sqrt(abs(b)**2/_tot)
        a = _a if a >= 0 else -1*_a
        b = 0 if abs(b) < TOL else b*(_b/abs(b))
        if a == 0 and b == 0: b = 1 # protection (a sets theta to π, b would be indiferent)
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
        return self.a*QuBit.KET_0 + self.b*QuBit.KET_1
    def state(self):
        return np.array([[self.a], [self.b]])
    def density_matrix(self):
        return QuBit.clean_matrix(self.state() @ np.transpose(np.conjugate(self.state())))
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
    def apply_X(self, to_self=True):
        """Also known as Pauli-X gate"""
        sigma_x = qg.X()
        if to_self:
            self.set_state(sigma_x @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_x @ self.state())
        return aux
    def apply_Y(self, to_self=True):
        """Also known as Pauli-Y gate"""
        sigma_y = qg.Y()
        if to_self:
            self.set_state(sigma_y @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_y @ self.state())
        return aux
    def apply_Z(self, to_self=True):
        """Also known as Pauli-Z gate"""
        sigma_z = qg.Z()
        if to_self:
            self.set_state(sigma_z @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_z @ self.state())
        return aux
    def apply_ID(self, to_self=True):
        sigma_id = qg.ID()
        if to_self:
            self.set_state(sigma_id @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_id @ self.state())
        return aux
    def apply_H(self, to_self=True):
        sigma_h = qg.H()
        if to_self:
            self.set_state(sigma_h @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_h @ self.state())
        return aux
    def apply_Rz_phi(self, phi, to_self=True):
        sigma_rz = qg.Rz_phi(phi)
        if to_self:
            self.set_state(sigma_rz @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_rz @ self.state())
        return aux
    def apply_S(self, to_self=True):
        sigma_s = qg.S()
        if to_self:
            self.set_state(sigma_s @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_s @ self.state())
        return aux
    def apply_S_dagger(self, to_self=True):
        sigma_s = qg.S_dagger()
        if to_self:
            self.set_state(sigma_s @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_s @ self.state())
        return aux
    def apply_T(self, to_self=True):
        sigma_t = qg.T()
        if to_self:
            self.set_state(sigma_t @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_t @ self.state())
        return aux
    def apply_T_dagger(self, to_self=True):
        sigma_t = qg.T_dagger()
        if to_self:
            self.set_state(sigma_t @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_t @ self.state())
        return aux
    def apply_Rx_phi(self, phi, to_self=True):
        sigma_rx = qg.Rx_phi(phi)
        if to_self:
            self.set_state(sigma_rx @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_rx @ self.state())
        return aux
    def apply_Ry_phi(self, phi, to_self=True):
        sigma_ry = qg.Ry_phi(phi)
        if to_self:
            self.set_state(sigma_ry @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_ry @ self.state())
        return aux
    def apply_sqNOT(self, to_self=True):
        sigma_sn = qg.sqNOT()
        if to_self:
            self.set_state(sigma_sn @ self.state())
            return
        aux = QuBit()
        aux.set_state(sigma_sn @ self.state())
        return aux
    def apply_RESET(self, to_self=True):
        """Use of this operation may make your code non-portable."""
        if to_self:
            self.set_ket('0')
            return
        aux = QuBit()
        aux.set_ket('0')
        return aux

def ndim_nested_loop(dimension, min_count=0, max_count=9):
    counter = [min_count]*dimension
    iterations = list()
    while True:
        iterations.append(list(reversed(counter)))
        counter[0] += 1
        for i in range(len(counter)-1):
            if counter[i] > max_count:
                counter[i] = min_count
                counter[i+1] += 1
        if counter[-1] > max_count:
            break
    return iterations

class QuRegister:
    available_gates = qg.argcounts()
    unary_gates = list(k for k, v in available_gates.items() if v <= 1)
    narity_gates = list(k for k, v in available_gates.items() if v > 1)
    def __init__(self, nr_qubits):
        if nr_qubits < 2:
            raise Exception("Consider using class Qubit for this")
        self.nr_qubits = nr_qubits
        s = list([1.] if i == 0 else [0.] for i in range(2**nr_qubits))
        self._state = np.array(s)
        self.msb_to_lsb = False
    def __getitem__(self, index):
        return self._state[index, 0]
    def msb_leftmost(self):
        """
        Feature for visualization only;
        Most Significant Bit (i.e. max subscripted bit) to the leftmost of register.
        msb_to_lsb=True, number 5 (4 bits) = 0101;
        most common computer science bit order, q4 to q0.
        """
        self.msb_to_lsb = True
    def msb_rightmost(self):
        """
        Feature for visualization only, reset to default;
        Most Significant Bit (i.e. max subscripted bit) to the rightmost of register.
        msb_to_lsb=False, number 5 (4 bits) = 1010;
        state vector bit order, q0 to q4, common to quantum computing operations.
        """
        self.msb_to_lsb = False
    def inverse_kron_product(self, tensor):
        """Reverse kronecker product known solution: Nearest Kronecker Product.
           Risk of swapping qubits order!!"""
        # https://math.stackexchange.com/questions/60399/method-to-reverse-a-kronecker-product/321424#321424
        m = tensor.shape[0]//2
        R = np.transpose(tensor.reshape(m, 2))
        if np.linalg.matrix_rank(R) != 1:
            print("Warning! Entanglement.")
        U, S, V = np.linalg.svd(R)
        sqrt_sigma = np.sqrt(S[0])
        tensor_a = sqrt_sigma*U[:, [0]]
        tensor_b = sqrt_sigma*np.transpose(V[[0], :])
        return tensor_a, tensor_b
    def get_qubit_collection(self):
        """NOT to be trusted; can swap qubit state order"""
        tensor = self.state()
        collection = [tensor]
        while True:
            t = collection.pop(-1)
            tensor_a, tensor_b = self.inverse_kron_product(t)
            collection += [tensor_a, tensor_b]
            if collection[-1].shape[0] == 2:
                break
        return collection
    def validate(self):
        state = self.state()
        if state.shape[1] != 1:
            raise Exception(f"Not a column vector: {state.shape}")
        a = state[0, 0]
        if abs(a.imag) < TOL: 
            state[0, 0] = a.real
        elif abs(a.imag) >= TOL: # ensure a \in R
            r1, phi1 = abs(a), cmath.phase(a)
            new_state = np.zeros(state.shape, dtype=state.dtype)
            new_state[0, 0] = r1
            for i in range(1, new_state.shape[0]):
                r2, phi2 = abs(state[i, 0]), cmath.phase(state[i, 0])
                new_state[i, 0] = r2*np.exp(1j*(phi2-phi1))
            _tot = np.sum(abs(new_state)**2)
            if _tot == 0: 
                _tot = 1
            new_state[:, 0] = np.sqrt(abs(new_state[:, 0])**2/_tot)
            state[0, 0] = new_state[0, 0] if state[0, 0] >= 0 else -1*new_state[0, 0]
            for i in range(1, new_state.shape[0]):
                if abs(new_state[i, 0]) < TOL:
                    state[i, 0] = 0.  
                else: 
                    state[i, 0] = state[i, 0]*(new_state[i, 0]/abs(state[i, 0]))
                if abs(state[i, 0].real) < TOL:
                    state[i, 0] -= state[i, 0].real
                if abs(state[i, 0].imag) < TOL:
                    state[i, 0] = state[i, 0].real
            if np.sum(abs(state)) < TOL: # protection (a sets theta to π, b would be indiferent)
                raise Exception(f"Please review state vector, all zeros found: {state}")
        self._state = state
    def state(self):
        """Return the statevector of the qubit"""
        return self._state
    def state_ket(self):
        return self.state()
    def state_bra(self):
        return np.transpose(np.conjugate(self.state()))
    def density_matrix(self):
        rho = self.state_ket() @ self.state_bra()
        rnk_rho = np.linalg.matrix_rank(rho)
        return QuBit.clean_matrix(rho)
    def partial_trace(self, qubit_2_keep):
        # https://gist.github.com/neversakura/d6a60b4bb2990d252e9e89e5629d5553
        """ Calculate the partial trace for qubit system
        Parameters
        ----------
        rho: np.ndarray
            Density matrix
        qubit_2_keep: list
            Index of qubit to be kept after taking the trace
        Returns
        -------
        rho_res: np.ndarray
            Density matrix after taking partial trace
        """
        rho = self.density_matrix()
        num_qubit = self.nr_qubits
        if type(qubit_2_keep) == int:
            qubit_2_keep = [qubit_2_keep]
        qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                    if i not in qubit_2_keep]
        minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
        minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                            for q, m in zip(qubit_axis, minus_factor)]
        rho_res = np.reshape(rho, [2, 2] * num_qubit)
        qubit_left = num_qubit - len(qubit_axis)
        for i, j in minus_qubit_axis:
            rho_res = np.trace(rho_res, axis1=i, axis2=j)
        if qubit_left > 1:
            rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)
        return rho_res
    def get_qubits(self):
        """This is a demonstration feature only, just a guess;
        Do not try to collapse these qubits individually;
        Initial experiments show that the presence of
        Bell states |Φ-❭ and |Ψ-❭ can mess with other qubits;
        Even when not, current states and recovered state
        from these qubits have significative differences;
        probabilities tends to keep up when no gate is applied, though"""
        qubits = list()
        for i in range(self.nr_qubits):
            dm_i = self.partial_trace(i)
            qubits.append(QuBit())
            qubits[i].set_state(dm_i @ QuBit.KET_0 + dm_i @ QuBit.KET_1)
        return qubits
    def qb_bin(self, num):
        b = bin(num)[2:]
        lb = len(b)
        bin_num = ("0"*(self.nr_qubits - lb) + b \
            if lb <= self.nr_qubits else b[-self.nr_qubits:])
        return bin_num[::-1] if self.msb_to_lsb else bin_num
    def probabilities(self):
        return np.abs(self.state())**2
    def assert_state(self):
        qubits = self.get_qubits()
        stt = qubits[0].state()
        for q in qubits[1:]:
            stt = np.kron(stt, q.state())
        aux = QuRegister(self.nr_qubits)
        aux.set(stt)
        state_self = self.state()
        state_aux = aux.state()
        for i in range(state_aux.shape[0]):
            if abs(state_aux[i, 0]) < TOL:
                state_aux[i, 0] = 0.
            if abs(state_aux[i, 0].real) < TOL:
                state_aux[i, 0] -= state_aux[i, 0].real
            if abs(state_aux[i, 0].imag) < TOL:
                state_aux[i, 0] = state_aux[i, 0].real
            if abs(state_aux[i, 0] - state_self[i, 0]) < TOL:
                print(f'* {i:03d}.) ok')
            else:
                print(f'* {i:03d}.)', stt[i, 0], state_self[i, 0], '!!!')
    def assert_probs(self):
        qubits = self.get_qubits()
        stt = qubits[0].state()
        for q in qubits[1:]:
            stt = np.kron(stt, q.state())
        pr_stt = np.abs(stt)**2
        probs = self.probabilities()
        for i in range(stt.shape[0]):
            if abs(pr_stt[i, 0]) < TOL:
                pr_stt[i, 0] = 0.
            if abs(pr_stt[i, 0].real) < TOL:
                pr_stt[i, 0] -= pr_stt[i, 0].real
            if abs(pr_stt[i, 0].imag) < TOL:
                pr_stt[i, 0] = pr_stt[i, 0].real
            if abs(pr_stt[i, 0] - probs[i, 0]) < TOL:
                print(f'* {i:03d}.) ok')
            else:
                print(f'* {i:03d}.)', pr_stt[i, 0], probs[i, 0], '!!!')
    def _table_probs_header(self, input_labels, sep_char='', end_char=''):
        def max_len(list_str):
            return max(list(len(x) for x in list_str))
        def w1pl(input_labels, sep_char, end_char, max_char=None):
            if max_char is None:
                max_char = max_len(input_labels)
            letters = list()
            for i in range(max_char):
                letters.append(' ' + sep_char.join(list(c[i]  if i < len(c) else ' ' for c in input_labels)) + ' ' + end_char)
            return letters
        labels = w1pl(input_labels, sep_char=sep_char, end_char=end_char)
        labels[0] += ' Prob'
        len_labels = max(len(l) for l in labels)
        h = '-' + '-'*len_labels + '\n'
        h += '\n'.join(labels) + '\n'
        h += '-' + '-'*len_labels + '\n'
        return h
    def table_prob(self, spaced=True):
        sep_char = ' ' if spaced else ''
        if self.msb_to_lsb:
            t = self._table_probs_header(list(f'q{i}' for i in reversed(range(self.nr_qubits))), 
                sep_char=sep_char, end_char='|')
        else:
            t = self._table_probs_header(list(f'q{i}' for i in range(self.nr_qubits)), 
                sep_char=sep_char, end_char='|')
        tb = list()
        for i in range(2**self.nr_qubits):
            state = list(c for c in f"{self.qb_bin(i)}")
            tb.append(' ' + sep_char.join(state) + f" | {100*abs(self[i])**2:.1f}%")
        t += '\n'.join(sorted(tb))
        return t
    def set(self, s):
        self._state = s
        self.validate()
    def init_from_qubits(self, *args):
        if len(args) == self.nr_qubits:
            qubits = list(QuBit() for _ in range(self.nr_qubits))
            for i, arg in enumerate(args):
                if type(arg) in [int, str]:
                    qubits[i].set_ket(arg)
                elif type(arg) in [tuple, list]:
                    qubits[i].set(*arg)
            s = qubits[0].state()
            for q in qubits[1:]:
                s = np.kron(s, q.state())
            self._state = s
            self.validate()
        else:
            raise Exception(f"Expecting {self.nr_qubits} qubits, not {len(args)} as informed")
    def measure(self):
        prob = list(p[0] for p in self.probabilities().tolist())
        chosen = np.random.choice(2**self.nr_qubits, size=None, p=prob, replace=True)
        return self.qb_bin(chosen)
    def histogram(self, outcomes, labels, plot=True, perc=False, title=""):
        h = dict()
        labels = sorted(labels)
        for l in labels:
            c = outcomes.count(l)
            h[l] = c
        if perc:
            total = len(outcomes)
            for i, l in enumerate(labels):
                h[l] /= total
        if plot:
            plt.rcParams.update({'font.size': 14})
            fig, ax = plt.subplots()
            ax.barh(labels, list(h[l] for l in labels), align='center')
            ax.invert_yaxis()
            ax.grid('on')
            plt.title("Probabilities for outcomes\n"+title)
            plt.xlabel("probability")
            plt.ylabel("outcome, " + ('leftmost MSB' if self.msb_to_lsb else 'rightmost MSB'))
            plt.show()
        return h
    def simulate(self, times=100, plot=True, perc=True, title=''):
        measurements = []
        for _ in range(times):
            measurements.append(self.measure())
        h = self.histogram(measurements, 
                sorted(list(self.qb_bin(s) for s in range(2**self.nr_qubits))), 
                plot, perc, title=title
            )
        return h
    def visualize(self, show=True, title=''):
        u = np.linspace(0, 2*np.pi, 21)
        v = np.linspace(0, np.pi, 11)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        ncols = int(np.ceil(np.sqrt(self.nr_qubits)))
        nrows = int(np.ceil(self.nr_qubits/ncols))
        qubits = self.get_qubits()
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw={'projection':'3d', 'title':''})
        axs = axs.reshape((nrows, ncols)) # in case of nrows or ncols = 1
        for i in range(nrows):
            for j in range(ncols):
                idx = (self.nr_qubits - 1) - (i*ncols + j) if self.msb_to_lsb else i*ncols + j
                if idx < self.nr_qubits:
                    axs[i][j].plot_wireframe(x, y, z, color='lightgray', linestyle=':')
                    axs[i][j].plot3D(np.cos(u), np.sin(u), np.zeros(u.shape), color='lightblue', linestyle='-.')
                    axs[i][j].plot3D(np.cos(u), np.zeros(u.shape), np.sin(u), color='lightblue', linestyle='-.')
                    axs[i][j].plot3D(np.zeros(u.shape), np.cos(u), np.sin(u), color='lightblue', linestyle='-.')
                    axs[i][j].plot3D([-1, 1], [0, 0], [0, 0], color='gray', linestyle='--')
                    axs[i][j].text(-1.05, 0, 0, '|$-$❭', 'x', horizontalalignment='right', fontweight='bold')
                    axs[i][j].text(1.05, 0, 0, '|$+$❭', 'x', horizontalalignment='left', fontweight='bold')
                    axs[i][j].plot3D([0, 0], [-1, 1], [0, 0], color='gray', linestyle='--')
                    axs[i][j].text(0, -1.05, 0, '|$-i$❭', 'y', horizontalalignment='right', fontweight='bold')
                    axs[i][j].text(0, 1.05, 0, '|$i$❭', 'y', horizontalalignment='left', fontweight='bold')
                    axs[i][j].plot3D([0, 0], [0, 0], [-1, 1], color='gray', linestyle='--')
                    axs[i][j].text(0, 0, -1.05, '|1❭', 'x', horizontalalignment='center', fontweight='bold')
                    axs[i][j].text(0, 0, 1.05, '|0❭', 'x', horizontalalignment='center', fontweight='bold')
                    axs[i][j].text(1.2, 0, 0, '$x$', 'x', horizontalalignment='left')
                    axs[i][j].text(0, 1.2, 0, '$y$', 'y', horizontalalignment='left')
                    axs[i][j].text(0, 0, 1.2, '$z$', 'x', horizontalalignment='center')
                    limits = np.array([getattr(axs[i][j], f'get_{axis}lim')() for axis in 'xyz'])
                    axs[i][j].set_box_aspect(np.ptp(limits, axis = 1), zoom=1.8)
                    q = qubits[idx]
                    axs[i][j].quiver(0, 0, 0, q.x, q.y, q.z, color='r', linewidth=2)
                    axs[i][j].legend(title_fontsize=14, title="$\\bf{"+f"q_{idx}"+"}$", handles=[])
                axs[i][j]._axis3don = False
        fig.suptitle(title + '\n' + ('leftmost MSB' if self.msb_to_lsb else 'rightmost MSB'), fontsize=12)
        if show:
            plt.show()
    def apply(self, *args):
        uargs = list()
        for arg in args:
            if arg[0].__name__ in QuRegister.unary_gates:
                if 'phi' in arg[0].__name__:
                    uargs.append((arg[2], arg[0], arg[1]))
                else:
                    uargs.append((arg[1], arg[0]))
        if uargs:
            self.set(qg.generate(*uargs, nr_qubits=self.nr_qubits) @ self.state())
        for arg in args:
            if arg[0].__name__ in QuRegister.narity_gates:
                gate = f"qg.{arg[0].__name__}(*{arg[1:]}, nr_qubits=self.nr_qubits)"
                circuit = eval(gate)
                self.set(circuit @ self.state())
    


if __name__ == '__main__':


    # qs = QuRegister(2)
    # for i in range(2):
    #     for j in range(2):
    #         qs.init_from_qubits(i, j)
    #         qs.set(qg.CNOT(1, 0) @ qs.state())
    #         print(i, j)
    #         print(qs.state())
    # print('***')
    # qs = QuRegister(2)
    # for i in range(2):
    #     for j in range(2):
    #         qs.init_from_qubits(i, j)
    #         qs.set(qg.CNOTrev(0, 1) @ qs.state())
    #         print(i, j)
    #         print(qs.state())
    # print('***')
    # qs = QuRegister(3)
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             qs.init_from_qubits(i, j, k)
    #             qs.set(qg.CSWAP(0, 1, 2) @ qs.state())
    #             print(i, j, k)
    #             print(qs.state())
    print('***')
    qs = QuRegister(4)
    # qs.init_from_qubits('i',0,(0.383,0.653-0.653j),1)
    print(qs.state().shape)
    qs.set(qg.generate((0, qg.H), (1, qg.X), (2, qg.Ry_phi, 3/4*np.pi), (3, qg.X), nr_qubits=4) @ qs.state())
    # print(qs.density_matrix())
    qs.set(qg.generate((0, qg.S), (2, qg.T_dagger), nr_qubits=4) @ qs.state())
    # qs.visualize(show=False, title='1. Setting')
    # # print((qg.entangle(1, 3, nr_qubits=4)@ qs.state()).shape)
    qs.set(qg.entangle(1, 3, nr_qubits=4) @ qs.state())
    # qs.visualize(show=False, title='2. Entangle 1-3')
    qs.set(qg.CNOT(0, 2, nr_qubits=4) @ qs.state())
    # qs.visualize(show=False, title='3. CNOT 0-2')
    # # # qs.set(qg.generate((0, qg.H), (1, qg.H), (2, qg.H), (3, qg.H), nr_qubits=4) @ qs.state())
    # # qs.set(qg.generate(nr_qubits=4, default=qg.H) @ qs.state())
    # qs.visualize(show=False, title='4. H applied all')
    print(qs.density_matrix())
    for i in range(qs.nr_qubits):
        print(i, '====')
        print(qs.partial_trace(i))
    # print(qs.state())
    # qs.get_qubits()
    # print('--')
    # qs.assert_state()
    # print('--')
    # qs.assert_probs()
    qs.simulate(10000)
    # qs.visualize()
    print('***')
    qr = QuRegister(4)
    qr.apply((qg.H, 0), (qg.X, 1), (qg.Ry_phi, 3/4*np.pi, 2), (qg.X, 3))
    qr.apply((qg.T_dagger, 2), (qg.entangle, 1, 3), (qg.S, 0))
    qr.apply((qg.CNOT, 0, 2))
    qr.simulate(10000)
    
    # print('***************')
    # qs.init_from_qubits(0,1,'i','+')
    # print(qs.state())
    # for i in range(4):
    #     print("#qb", i, 0)
    #     test = qg.generate((i, qg.ket0_bra0), nr_qubits=4)@qs.state()
    #     print(test, '\n', np.sum(test), abs(np.sum(test)))
    #     print("#qb", i, 1)
    #     test = qg.generate((i, qg.ket1_bra1), nr_qubits=4)@qs.state()
    #     print(test, '\n', np.sum(test), abs(np.sum(test)))

    # qs = MultiQubits(3)
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             qs.init_from_qubits(i, j, k)
    #             qs.CCNOT_gate(0, 1, 2)
    #             print(i, j, k)
    #             print(qs.state())


    
    # # # q = Qubit()
    # # # q.set(1,1)
    # # # print(q.state())
    # # # q.simulate(1000)
    # # qs = MultiQubits(3)
    # # qs.init_from_qubits(
    # #     # |0❭     |1❭
    # #     # (1,0),
    # #     # (1,1),
    # #     # (1,0),
    # #     # (1,      1  ),
    # #     # (1,      1  ),
    # #     # (1,      1  )
    # #     (.3-.3j, 0.8   ),
    # #     (1,      2     ),
    # #     (0.8,    .3-.3j)
    # # )
    # # print(qs.table_prob())
    # # # qs.set(0,(1,1))
    # # s = qs.state()
    # # print(s.size)
    # # print(s)

    # # print(qs.bin(1024))
    # # # print('      00     01     10     11')
    # # # for i in range(2):
    # # #     for j in range(2):
    # # #         qs.set(i, (3,3))
    # # #         s = qs.state()
    # # #         print(i, 's', f"{s.tolist()}'")

    # # # for _ in range(20):
    # # #     print(qs.measure(), end=', ')
    # # # print(" ")

    # # qs.simulate(1000)

    # # import time
    # qs = MultiQubits(6)
    # qs.init_from_qubits(*list(c for c in '01+i10'))
    # # qs.Hn_gate()
    # # qs.apply_gates((0, 'ID'), (1, 'X'))
    # # qs.apply_gates((0, 'X'))
    # # qs.apply_gates((0, 'H'))
    # # print(qs.state())
    # # qs.simulate(100)

    # aux = qs.get_qubits()
    # for a in aux:
    #     print(a)
    # k = aux[0]
    # for a in aux[1:]:
    #     k = np.kron(k, a)
    # ps = MultiQubits(6)
    # ps.set(k)
    # # qs.simulate()
    # # ps.simulate()

    # for x, y in zip(qs.state()[:,0].tolist(), ps.state()[:,0].tolist()):
    #     if abs(x.real - y.real) < TOL and abs(x.imag - y.imag) < TOL:
    #         print('same')
    #     else:
    #         print(f'NOT same: {x} {y}')

    # # q = Qubit()
    # # q.H_gate()
    # # q.Z_gate()
    # # q.H_gate()
    # # print('>', q.measure())
