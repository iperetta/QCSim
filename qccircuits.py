import qubit as qb
from qugate import QuGates as qg

# qr = qb.QuRegister(3)

# for i in range(2):
#     for j in range(2):
#         qr.init_from_qubits(i, j, 0)
#         qr.set()

qbit = qb.QuRegister(2) # 0 0 
# qbit.apply((qg.X, 0)) # 1 0 
qbit.apply((qg.H, 0))
qbit.apply((qg.CNOT, 0, 1))
print(qbit.state())
print("")
print(qbit.table_prob())
qbit.simulate(100000)