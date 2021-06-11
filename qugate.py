# Attempt to organize code
import numpy as np

class QuGates:
    TOL = 2*np.finfo(float).eps
    @classmethod
    def list_gates(cls, arity=None):
        gates = list(n for n in dir(QuGates) if not n.startswith('_'))
        for n in ['clean_matrix', 'generate', 'TOL', 'list_gates', 'argcounts']:
            gates.remove(n)
        if arity is None:
            return gates
        elif arity == 1:
            return list(g for g in gates if not ('C' in g or 'SWAP' in g or 'entagle' in g))
        return list(g for g in gates if g.endswith('n'))
    @staticmethod
    def argcounts():
        return dict((g, eval(f"QuGates.{g}.__code__.co_argcount")) for g in QuGates.list_gates())
    @staticmethod
    def clean_matrix(a):
        m, n = a.shape
        for i in range(m):
            for j in range(n):
                if abs(a[i, j]) < QuGates.TOL:
                    a[i, j] = 0.
                if abs(a[i, j].real) < QuGates.TOL:
                    a[i, j] -= a[i, j].real # discard real part
                if abs(a[i, j].imag) < QuGates.TOL:
                    a[i, j] = a[i, j].real  # keep real part only
        return a
    # Unary gates
    @staticmethod
    def zero():
        return np.zeros((2, 2))
    @staticmethod
    def ket0_bra0(): 
        return np.array([[1., 0.], [0., 0.]])
    @staticmethod
    def ket0_bra1(): 
        return np.array([[0., 1.], [0., 0.]])
    @staticmethod
    def ket1_bra0(): 
        return np.array([[0., 0.], [1., 0.]])
    @staticmethod
    def ket1_bra1(): 
        return np.array([[0., 0.], [0., 1.]])
    @staticmethod
    def ID(): 
        return np.array([[1., 0.], [0., 1.]])
    @staticmethod
    def X(): # Pauli-X gate
        return np.array([[0., 1.], [1., 0.]])
    @staticmethod
    def Y(): # Pauli-Y gate
        return 1j*np.array([[0., -1.], [1., 0.]])
    @staticmethod
    def Z(): # Pauli-Z gate
        return np.array([[1., 0.], [0., -1.]])
    @staticmethod
    def H(): # Hadamard gate
        return 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
    @staticmethod
    def Rx_phi(phi): 
        return QuGates.clean_matrix(np.array([
            [np.cos(phi/2), -np.sin(phi/2)*1j], 
            [-np.sin(phi/2)*1j, np.cos(phi/2)]
        ]))
    @staticmethod
    def Ry_phi(phi): 
        return QuGates.clean_matrix(np.array([
            [np.cos(phi/2), -np.sin(phi/2)], 
            [np.sin(phi/2), np.cos(phi/2)]
        ]))
    @staticmethod
    def Rz_phi(phi): 
        return QuGates.clean_matrix(np.array([[1, 0], [0, np.exp(phi*1j)]]))
    @staticmethod
    def S(): 
        return QuGates.Rz_phi(np.pi/2)
    @staticmethod
    def S_dagger(): 
        return QuGates.Rz_phi(3*np.pi/2)
    @staticmethod
    def T(): 
        return QuGates.Rz_phi(np.pi/4)
    @staticmethod
    def T_dagger(): 
        return QuGates.Rz_phi(7*np.pi/4)
    @staticmethod
    def sqNOT():
        return 0.5*np.array([
            [1 + 1j, 1 - 1j], 
            [1 - 1j, 1 + 1j]
        ])
    # Support for n-arity gates
    @staticmethod
    def generate(*args, nr_qubits=2, default=None):
        """Args is sequence of tuples each of (qubit_idx, gate) or (qubit_idx, gate, phi)"""
        sigma_id = QuGates.ID() if default is None else default()
        sigmas = list(sigma_id for _ in range(nr_qubits))
        for arg in args:
            qubit_idx = arg[0]
            phi = None if len(arg) == 2 else arg[2]
            gate = arg[1]
            if 'phi' in gate.__name__ and phi is None:
                raise Exception(f"Gate {gate} needs 'phi' parameter")
            sigmas[qubit_idx] = gate() if phi is None else gate(phi)
        qg = sigmas[0]
        for i in range(1, len(sigmas)):
            qg = np.kron(qg, sigmas[i])
        return qg
    # "N-arity" 2-qubits gates
    @staticmethod
    def SWAP(qb_idx_a=0, qb_idx_b=1, nr_qubits=2):
        if max(qb_idx_a, qb_idx_b) >= nr_qubits:
            nr_qubits = max(qb_idx_a, qb_idx_b) + 1
        sigma = QuGates.generate(
            (qb_idx_a, QuGates.ket0_bra0), (qb_idx_b, QuGates.ket0_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (qb_idx_a, QuGates.ket0_bra1), (qb_idx_b, QuGates.ket1_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (qb_idx_a, QuGates.ket1_bra0), (qb_idx_b, QuGates.ket0_bra1),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (qb_idx_a, QuGates.ket1_bra1), (qb_idx_b, QuGates.ket1_bra1),
            nr_qubits=nr_qubits
        )
        return QuGates.clean_matrix(sigma)
    @staticmethod
    def CU(unary_gate, ctrl_idx=0, fx_idx=1, nr_qubits=2):
        if max(ctrl_idx, fx_idx) >= nr_qubits:
            nr_qubits = max(ctrl_idx, fx_idx) + 1
        phi = None
        if type(unary_gate) is tuple:
            unary_gate, phi = unary_gate
        sigma = QuGates.generate(
            (ctrl_idx, QuGates.ket0_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx, QuGates.ket1_bra1), (fx_idx, unary_gate, phi),
            nr_qubits=nr_qubits
        )
        return QuGates.clean_matrix(sigma)
    @staticmethod
    def CNOT(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU(QuGates.X, ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def sqCNOT(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU(QuGates.sqNOT, ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CX(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU(QuGates.X, ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CY(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU(QuGates.Y, ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CZ(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU(QuGates.Z, ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CsqNOT(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU(QuGates.sqNOT, ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CRx_phi(phi, ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU((QuGates.Rx_phi, phi), ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CRy_phi(phi, ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU((QuGates.Ry_phi, phi), ctrl_idx, fx_idx, nr_qubits)
    @staticmethod
    def CRz_phi(phi, ctrl_idx=0, fx_idx=1, nr_qubits=2):
        return QuGates.CU((QuGates.Rz_phi, phi), ctrl_idx, fx_idx, nr_qubits)
    # "N-arity" 3-qubits gates
    @staticmethod
    def CSWAP(ctrl_idx=0, qb_idx_a=1, qb_idx_b=2, nr_qubits=3): # Fredkin gate
        if max(ctrl_idx, qb_idx_a, qb_idx_b) >= nr_qubits:
            nr_qubits = max(ctrl_idx, qb_idx_a, qb_idx_b) + 1
        sigma = QuGates.generate(
            (ctrl_idx, QuGates.ket0_bra0), nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx, QuGates.ket1_bra1),
            (qb_idx_a, QuGates.ket0_bra0), (qb_idx_b, QuGates.ket0_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx, QuGates.ket1_bra1),
            (qb_idx_a, QuGates.ket0_bra1), (qb_idx_b, QuGates.ket1_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx, QuGates.ket1_bra1),
            (qb_idx_a, QuGates.ket1_bra0), (qb_idx_b, QuGates.ket0_bra1),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx, QuGates.ket1_bra1),
            (qb_idx_a, QuGates.ket1_bra1), (qb_idx_b, QuGates.ket1_bra1),
            nr_qubits=nr_qubits
        )
        return QuGates.clean_matrix(sigma)
    @staticmethod
    def CCNOT(ctrl_idx_a=0, ctrl_idx_b=1, fx_idx=2, nr_qubits=3): # Toffoli gate
        if max(ctrl_idx_a, ctrl_idx_b, fx_idx) >= nr_qubits:
            nr_qubits = max(ctrl_idx_a, ctrl_idx_b, fx_idx) + 1
        sigma = QuGates.generate(
            (ctrl_idx_a, QuGates.ket0_bra0), 
            (ctrl_idx_b, QuGates.ket0_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx_a, QuGates.ket0_bra0), 
            (ctrl_idx_b, QuGates.ket1_bra1),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx_a, QuGates.ket1_bra1), 
            (ctrl_idx_b, QuGates.ket0_bra0),
            nr_qubits=nr_qubits
        )
        sigma = sigma + QuGates.generate(
            (ctrl_idx_a, QuGates.ket1_bra1), 
            (ctrl_idx_b, QuGates.ket1_bra1),
            (fx_idx, QuGates.X),
            nr_qubits=nr_qubits
        )
        return QuGates.clean_matrix(sigma)
    # Circuits to achieve effects
    @staticmethod
    def sqSWAP(qb_idx_a=0, qb_idx_b=1, nr_qubits=2):
        if max(qb_idx_a, qb_idx_b) >= nr_qubits:
            nr_qubits = max(qb_idx_a, qb_idx_b) + 1
        sigma = QuGates.CNOT(qb_idx_a, qb_idx_b, nr_qubits)
        sigma = QuGates.sqCNOT(qb_idx_b, qb_idx_a, nr_qubits) @ sigma
        sigma = QuGates.CNOT(qb_idx_a, qb_idx_b, nr_qubits) @ sigma
        return QuGates.clean_matrix(sigma)
    @staticmethod
    def CNOTrev(ctrl_idx=0, fx_idx=1, nr_qubits=2):
        if max(ctrl_idx, fx_idx) >= nr_qubits:
            nr_qubits = max(ctrl_idx, fx_idx) + 1
        sigma = QuGates.generate(
            (ctrl_idx, QuGates.H), (fx_idx, QuGates.H),
            nr_qubits=nr_qubits
        )
        sigma = QuGates.CNOT(ctrl_idx, fx_idx, nr_qubits) \
            @ sigma
        sigma = QuGates.generate(
            (ctrl_idx, QuGates.H), (fx_idx, QuGates.H),
            nr_qubits=nr_qubits
        ) @ sigma
        return QuGates.clean_matrix(sigma)
    @staticmethod
    def entangle(qb_idx_a=0, qb_idx_b=1, nr_qubits=2):
        """
        input :: Bell state;
        |00❭  :: |Φ+❭;
        |01❭  :: |Ψ+❭;
        |10❭  :: |Ψ-❭;
        |11❭  :: |Φ-❭
        """
        if max(qb_idx_a, qb_idx_b) >= nr_qubits:
            nr_qubits = max(qb_idx_a, qb_idx_b) + 1
        sigma = QuGates.generate(
            (qb_idx_a, QuGates.H), nr_qubits=nr_qubits
        )
        sigma = QuGates.CNOT(
            qb_idx_a, qb_idx_b, nr_qubits=nr_qubits
        ) @ sigma
        return QuGates.clean_matrix(sigma)
    # @staticmethod
    # def measure(statevector, qb_id=None, nr_qubits=1):
    #     probs = np.abs(statevector)**2


if __name__ == '__main__':
    gates = QuGates.list_gates()
    print(gates)
    print(QuGates.argcounts())
    gates.remove('CU')
    print("Gate CU is being tested by all gates which depend on it.")
    print("Note that tests consider minimum number of qubits only:")
    for g in gates:
        print('=', g, '='*(30 - len(g)))
        matrix = eval(f"QuGates.{g}(np.pi/3)") if 'phi' in g else eval(f"QuGates.{g}()")
        print(matrix)