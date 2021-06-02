# Attempt to organize code
import numpy as np

class QuGates:
    TOL = 2*np.finfo(float).eps
    @staticmethod
    def clean_matrix(a):
        m, n = a.shape
        for i in range(m):
            for j in range(n):
                if abs(a[i, j]) < QuGates.TOL:
                    a[i, j] = 0.
        return a
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
        return np.array([
            [np.cos(phi/2), -np.sin(phi/2)*1j], 
            [-np.sin(phi/2)*1j, np.cos(phi/2)]
        ])
    @staticmethod
    def Ry_phi(phi): 
        return np.array([
            [np.cos(phi/2), -np.sin(phi/2)], 
            [np.sin(phi/2), np.cos(phi/2)]
        ])
    @staticmethod
    def Rz_phi(phi): 
        return np.array([[1, 0], [0, np.exp(phi*1j)]])
    @staticmethod
    def S(): 
        return QuGates.Rz_phi(np.pi/2)
    @staticmethod
    def S_cross(): 
        return QuGates.Rz_phi(3*np.pi/2)
    @staticmethod
    def T(): 
        return QuGates.Rz_phi(np.pi/4)
    @staticmethod
    def T_cross(): 
        return QuGates.Rz_phi(7*np.pi/4)
    @staticmethod
    def sqNOT():
        return 0.5*np.array([
            [1 + 1j, 1 - 1j], 
            [1 - 1j, 1 + 1j]
        ])
    @staticmethod
    def generate(*args, nr_qubits=2):
        """Args is sequence of tuples each of (qubit_idx, gate) or (qubit_idx, gate, phi)"""
        sigma_id = QuGates.ID()
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
    @staticmethod
    def SWAP(qb_idx_a=0, qb_idx_b=1, nr_qubits=2):
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
    def CNOTrev(ctrl_idx=0, fx_idx=1, nr_qubits=2):
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
    @staticmethod
    def CSWAP(ctrl_idx=0, qb_idx_a=1, qb_idx_b=2, nr_qubits=3): # Fredkin gate
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
    @staticmethod
    def entangle(qb_idx_a=0, qb_idx_b=1, nr_qubits=2):
        """
        input :: Bell state;
        |00❭  :: |Φ+❭;
        |01❭  :: |Ψ+❭;
        |10❭  :: |Φ-❭;
        |11❭  :: |Ψ-❭
        """
        sigma = QuGates.generate(
            (qb_idx_a, QuGates.H), nr_qubits=nr_qubits
        )
        sigma = QuGates.CNOT(
            qb_idx_a, qb_idx_b, nr_qubits=nr_qubits
        ) @ sigma
        return QuGates.clean_matrix(sigma)





# class QuGate:
#     def __init__(self, nr_qubits=1):
#         self.nr_qubits = nr_qubits
#     def unitary_gate_ID(self):
#         return np.array([[1., 0.], [0., 1.]])
#     def n_arity_gate_matrix(self, *args):
#         """Args is sequence of tuples each of (qubit_idx, gate) or (qubit_idx, gate, phi)"""
#         sigma_id = self.unitary_gate_ID()
#         sigmas = list(sigma_id for _ in range(self.nr_qubits))
#         for arg in args:
#             gate = arg[0]
#             qubit_idx = arg[1]
#             phi = None if len(arg) == 2 else arg[2]
#             if 'phi' in gate and phi is None:
#                 raise Exception(f"Gate {gate} needs 'phi' parameter")
#             sigmas[qubit_idx] = QUGATES[gate]
#         qgt = sigmas[0]
#         for i in range(1, self.nr_qubits):
#             qgt = np.kron(qgt, sigmas[i])
#         return qgt
#     def generate(self, *args, nr_qubits=None):
#         """
#         generate(('H', 0))
#         generate(('X', 1), ('Y', 2))
#         generate(('CNOT', (2, 3)))
#         generate(('CCNOT', (1, 3, 2)))
#         generate((('Rx_phi', np.pi/5), 0))
#         generate((('CRx_phi', np.pi/5), (2, 3)))
#         """
#         if nr_qubits is None:
#             nr_qubits = self.nr_qubits
#         for arg in args:
#             head, tail = arg
#             if type(head) is tuple:
#                 gate_lbl = head[0]
#                 phi = head[1]
#             else:
#                 gate_lbl = head
#                 phi = None
#             if type(tail) is int:
#                 qb_idx = [tail]
#             else:
#                 qb_idx = tail

if __name__ == '__main__':
    from qubit import QuRegister
    # qr = QuRegister(2)
    # for i in range(2):
    #     for j in range(2):
    #         qr.init_from_qubits(i, j)
    #         print(i, j, '===================')
    #         print(qr.state())
    #         print(NArityQG.SWAP(0, 1) @ qr.state())
    print(QuGates.SWAP(0, 1))
    print(QuGates.CNOT(1, 0))
    print(QuGates.CNOTrev(0, 1))
    print(QuGates.CZ(0, 1))
    print(QuGates.CRx_phi(np.pi, 0, 1))
    print(QuGates.CSWAP(0, 1, 2))
    print(QuGates.CCNOT(0, 1, 2))
    print('****')
    qr = QuRegister(2)
    for i in range(2):
        for j in range(2):
            qr.init_from_qubits(i, j)
            print(i, j, '===================')
            print(QuGates.entangle(0, 1) @ qr.state())