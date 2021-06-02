import numpy as np

phi_zfun = lambda phi : np.array([
    [1, 0], 
    [0, np.exp(phi*1j)]
])

QUGATES = {
    'X': np.array([[0., 1.], [1., 0.]]),
    'Y': 1j*np.array([[0., -1.], [1., 0.]]),
    'Z': np.array([[1., 0.], [0., -1.]]),
    'ID': np.array([[1., 0.], [0., 1.]]),
    'H': 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]),
    'Rx_phi': lambda phi : np.array([
        [np.cos(phi/2), -np.sin(phi/2)*1j], 
        [-np.sin(phi/2)*1j, np.cos(phi/2)]
    ]),
    'Ry_phi': lambda phi : np.array([
        [np.cos(phi/2), -np.sin(phi/2)], 
        [np.sin(phi/2), np.cos(phi/2)]
    ]),
    'Rz_phi': lambda phi : phi_zfun(phi),
    'S':  phi_zfun(np.pi/2),
    'S+': phi_zfun(3*np.pi/2),
    'T':  phi_zfun(np.pi/4),
    'T+': phi_zfun(7*np.pi/4),
    'sqNOT': 0.5*np.array([
        [1 + 1j, 1 - 1j], 
        [1 - 1j, 1 + 1j]
    ]),
    '|0><0|': np.array([[1., 0.], [0., 0.]]),
    '|0><1|': np.array([[0., 1.], [0., 0.]]),
    '|1><0|': np.array([[0., 0.], [1., 0.]]),
    '|1><1|': np.array([[0., 0.], [0., 1.]]),
}

class MQuGates:
    def __init__(self, nr_qubits):
        self.nr_qubits = nr_qubits
        self.gates = list(g[:-len('_gate')] for g in dir(MQuGates) if g.endswith('_gate'))
    def get(self, gate):
        if gate in self.gates:
            return eval(f'self.{gate}_gate')
        raise Exception(f"Gate {gate} inexistent")
    def multi_gate_sigma(self, *args):
        """Args is sequence of tuples each of (qubit_idx, gate) or (qubit_idx, gate, phi)"""
        sigma_id = QUGATES['ID']
        sigmas = list(sigma_id for _ in range(self.nr_qubits))
        for arg in args:
            qubit_idx = arg[0]
            gate = arg[1]
            phi = None if len(arg) == 2 else arg[2]
            if 'phi' in gate and phi is None:
                raise Exception(f"Gate {gate} needs 'phi' parameter")
            sigmas[qubit_idx] = QUGATES[gate]
        qg = sigmas[0]
        for i in range(1, self.nr_qubits):
            qg = np.kron(qg, sigmas[i])
        return qg
    def Hn_gate(self):
        args = list((i, 'H') for i in range(self.nr_qubits))
        return self.multi_gate_sigma(*args)
    def SWAP_gate(self, qba_idx, qbb_idx):
        # https://quantumcomputing.stackexchange.com/a/5192/16056 # The Algorithmic Method
        # https://quantumcomputing.stackexchange.com/a/9182/16056 
        if qba_idx == qbb_idx:
            raise Exception("Review SWAP for same qubit")
        args = list((i, 'ID') for i in range(self.nr_qubits))
        args[qba_idx] = (qba_idx, '|0><0|')
        args[qbb_idx] = (qbb_idx, '|0><0|')
        sigma_gate_00 = self.multi_gate_sigma(*args)
        args[qba_idx] = (qba_idx, '|0><1|')
        args[qbb_idx] = (qbb_idx, '|1><0|')
        sigma_gate_01 = self.multi_gate_sigma(*args)
        args[qba_idx] = (qba_idx, '|1><0|')
        args[qbb_idx] = (qbb_idx, '|0><1|')
        sigma_gate_10 = self.multi_gate_sigma(*args)
        args[qba_idx] = (qba_idx, '|1><1|')
        args[qbb_idx] = (qbb_idx, '|1><1|')
        sigma_gate_11 = self.multi_gate_sigma(*args)
        return sigma_gate_00 + sigma_gate_01 + sigma_gate_10 + sigma_gate_11
    def CU_gate(self, flip_gate, control, flip, phi=None):
        # https://quantumcomputing.stackexchange.com/questions/5409/composing-the-cnot-gate-as-a-tensor-product-of-two-level-matrices
        # https://quantumcomputing.stackexchange.com/a/5192/16056 # The Algorithmic Method
        if control == flip:
            raise Exception("Qubit cannot be controled by itself")
        args = list((i, 'ID') for i in range(self.nr_qubits))
        args[control] = (control, '|0><0|')
        args[flip] = (flip, 'ID')
        sigma_gate_0 = self.multi_gate_sigma(*args)
        args[control] = (control, '|1><1|')
        args[flip] = (flip, flip_gate) if not 'phi' in flip_gate else (flip, flip_gate, phi)
        sigma_gate_1 = self.multi_gate_sigma(*args)
        return sigma_gate_0 + sigma_gate_1
    def CNOT_gate(self, control, flip):
        """Inform control and flip qubits indices (in this order)"""
        return self.CU_gate('X', control, flip)
    def CNOTrev_gate(self, control, flip):
        """Inform control and flip qubits indices (in this order)"""
        sigma = self.multi_gate_sigma((control, 'H'), (flip, 'H'))
        sigma = self.CNOT_gate(control, flip) @ sigma
        sigma = self.multi_gate_sigma((control, 'H'), (flip, 'H')) @ sigma
        return sigma
    def CX_gate(self, control, flip):
        """Inform control and flip qubits indices (in this order)"""
        return self.CU_gate('X', control, flip)
    def CY_gate(self, control, flip):
        """Inform control and flip qubits indices (in this order)"""
        return self.CU_gate('Y', control, flip)
    def CZ_gate(self, control, flip):
        """Inform control and flip qubits indices (in this order)"""
        return self.CU_gate('Z', control, flip)
    def CsqNOT_gate(self, control, flip):
        """Inform control and flip qubits indices (in this order)"""
        return self.CU_gate('sqNOT', control, flip)
    def CRx_phi_gate(self, phi, control, flip):
        """Inform phi angle and both control and flip qubits indices (in this order)"""
        return self.CU_gate('Rx_phi', control, flip, phi)
    def CRy_phi_gate(self, phi, control, flip):
        """Inform phi angle and both control and flip qubits indices (in this order)"""
        return self.CU_gate('Ry_phi', control, flip, phi)
    def CRz_phi_gate(self, phi, control, flip):
        """Inform phi angle and both control and flip qubits indices (in this order)"""
        return self.CU_gate('Rz_phi', control, flip, phi)
    def CCNOT_gate(self, ctrl1, ctrl2, flip):
        """Also known as Toffoli gate"""
        if ctrl1 == flip or ctrl2 == flip:
            raise Exception("Qubit cannot be controled by itself")
        if ctrl1 == ctrl2:
            raise Exception("Consider using CNOT gate")
        args = list((i, 'ID') for i in range(self.nr_qubits))
        #cases not for action
        args[ctrl1] = (ctrl1, '|0><0|')
        args[ctrl2] = (ctrl2, '|0><0|')
        args[flip] = (flip, 'ID')
        sigma_gate_00 = self.multi_gate_sigma(*args)
        args[ctrl1] = (ctrl1, '|0><0|')
        args[ctrl2] = (ctrl2, '|1><1|')
        args[flip] = (flip, 'ID')
        sigma_gate_01 = self.multi_gate_sigma(*args)
        args[ctrl1] = (ctrl1, '|1><1|')
        args[ctrl2] = (ctrl2, '|0><0|')
        args[flip] = (flip, 'ID')
        sigma_gate_10 = self.multi_gate_sigma(*args)
        #case for action
        args[ctrl1] = (ctrl1, '|1><1|')
        args[ctrl2] = (ctrl2, '|1><1|')
        args[flip] = (flip, 'X')
        sigma_gate_11 = self.multi_gate_sigma(*args)
        return sigma_gate_00 + sigma_gate_01 + sigma_gate_10 + sigma_gate_11
    def CSWAP_gate(self, control, qba_idx, qbb_idx):
        """Also known as Fredkin gate"""
        if control == qba_idx or control == qbb_idx:
            raise Exception("Qubit cannot be controled by itself")
        if qba_idx == qbb_idx:
            raise Exception("No swap to control here")
        args = list((i, 'ID') for i in range(self.nr_qubits))
        args[control] = (control, '|0><0|')
        sigma_gate_c0_XX = self.multi_gate_sigma(*args)
        args[control] = (control, '|1><1|')
        args[qba_idx] = (qba_idx, '|0><0|')
        args[qbb_idx] = (qbb_idx, '|0><0|')
        sigma_gate_c1_00 = self.multi_gate_sigma(*args)
        args[control] = (control, '|1><1|')
        args[qba_idx] = (qba_idx, '|0><1|')
        args[qbb_idx] = (qbb_idx, '|1><0|')
        sigma_gate_c1_01 = self.multi_gate_sigma(*args)
        args[control] = (control, '|1><1|')
        args[qba_idx] = (qba_idx, '|1><0|')
        args[qbb_idx] = (qbb_idx, '|0><1|')
        sigma_gate_c1_10 = self.multi_gate_sigma(*args)
        args[control] = (control, '|1><1|')
        args[qba_idx] = (qba_idx, '|1><1|')
        args[qbb_idx] = (qbb_idx, '|1><1|')
        sigma_gate_c1_11 = self.multi_gate_sigma(*args)
        return sigma_gate_c0_XX + sigma_gate_c1_00 + sigma_gate_c1_01 + \
            sigma_gate_c1_10 + sigma_gate_c1_11
    def entanglement(self, qba_idx, qbb_idx):
        """
        input :: Bell state;
        |00❭  :: |Φ+❭;
        |01❭  :: |Ψ+❭;
        |10❭  :: |Φ-❭;
        |11❭  :: |Ψ-❭
        """
        sigma = self.multi_gate_sigma((qba_idx, 'H'))
        sigma = self.CNOT_gate(qba_idx, qbb_idx) @ sigma
        return sigma

# print(list(g[:-len('_gate')] for g in dir() if g.endswith('_gate')))
# MQUGATES = dict()
