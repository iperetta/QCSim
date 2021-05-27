import numpy as np

phi_zfun = lambda phi : np.array([
    [1, 0], 
    [0, np.exp(phi*1j)]
])

QGATES = {
    'X': np.array([[0, 1], [1, 0]]),
    'Y': 1j*np.array([[0, -1], [1, 0]]),
    'Z': np.array([[1, 0], [0, -1]]),
    'ID': np.array([[1, 0], [0, 1]]),
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
}