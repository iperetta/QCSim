import qubit as qb
from qugate import QuGates as qg

# 2-qubits Register inicialization 00-01-10-11
for j in range(2):
    for i in range(2):
        qr = qb.QuRegister(2)
        if i == 1:
            qr.apply((qg.X, 0))
        if j == 1:
            qr.apply((qg.X, 1))
        qr.visualize(show=False)
        qr.simulate(10000, title=f"|{i}{j}❭")

# 2-qubits entanglement, 4 Bell states:
# |00❭ -> |Φ+❭; |01❭ -> |Ψ+❭; |10❭ -> |Ψ-❭; |11❭ -> |Φ-❭
for j in range(2):
    for i in range(2):
        qr = qb.QuRegister(2)
        if i == 1:
            qr.apply((qg.X, 0))
        if j == 1:
            qr.apply((qg.X, 1))
        qr.apply( # entanglement
            (qg.H, 0),
            (qg.CNOT, 0, 1),
        )
        qr.visualize(show=False) 
        msg = ''
        if i == 0 and j == 0: msg = "|Φ+❭"
        if i == 0 and j == 1: msg = "|Ψ+❭"
        if i == 1 and j == 0: msg = "|Ψ-❭"
        if i == 1 and j == 1: msg = "|Φ-❭"
        qr.simulate(10000, title=f"|{i}{j}❭ ->"+msg)

# qbit = qb.QuRegister(2) # 0 0 
# # qbit.apply((qg.X, 0)) # 1 0 
# # qbit.apply((qg.X, 1)) # 1 0 
# qbit.apply((qg.H, 0))
# qbit.apply((qg.S, 0))
# qbit.apply((qg.H, 1))
# qbit.apply((qg.S, 1))
# print(qbit.state())
# qbit.visualize()
# qbit.apply((qg.CsqNOT, 0, 1))
# print(qbit.state())
# print("")
# print(qbit.table_prob())
# # qbit.simulate(100000)
# print(qbit.partial_trace(0))
# print(qbit.partial_trace(1))
# qbit.visualize()
# qbit.assert_state()
# qbit.assert_probs()