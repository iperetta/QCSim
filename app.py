import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
from qubit import Qubit

# Define qubit
q = Qubit()
p = Qubit()

# To plot qubits
def plot_qubits(*args, **kwargs):
    def repel_from_center(x, y, z, m=0.1):
        return x + (-m if x < 0 else m), \
            y + (-m if y < 0 else m), \
            z + (-m if z < 0 else m)
    def bloch_sphere(ax):
        ax.clear()
        u = np.linspace(0, 2*np.pi, 21)
        v = np.linspace(0, np.pi, 11)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x, y, z, color='gray', linestyle=':')
        ax.plot3D([-1, 1], [0, 0], [0, 0], color='k', linestyle='--')
        ax.text(-1.05, 0, 0, '|$-$❭', 'x', horizontalalignment='right', fontweight='bold')
        ax.text(1.05, 0, 0, '|$+$❭', 'x', horizontalalignment='left', fontweight='bold')
        ax.plot3D([0, 0], [-1, 1], [0, 0], color='k', linestyle='--')
        ax.text(0, -1.05, 0, '|$-i$❭', 'y', horizontalalignment='right', fontweight='bold')
        ax.text(0, 1.05, 0, '|$i$❭', 'y', horizontalalignment='left', fontweight='bold')
        ax.plot3D([0, 0], [0, 0], [-1, 1], color='k', linestyle='--')
        ax.text(0, 0, -1.05, '|1❭', 'x', horizontalalignment='center', fontweight='bold')
        ax.text(0, 0, 1.05, '|0❭', 'x', horizontalalignment='center', fontweight='bold')
        ax.text(1.2, 0, 0, '$x$', 'x', horizontalalignment='left')
        ax.text(0, 1.2, 0, '$y$', 'y', horizontalalignment='left')
        ax.text(0, 0, 1.2, '$z$', 'x', horizontalalignment='center')
        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis = 1), zoom=1.8)
        ax._axis3don = False
        return ax
    def draw_theta(ax, c='k'):
        t = np.linspace(0.1, np.pi, 31)
        x = 0.1 * np.sin(t)
        y = np.zeros(np.size(x))
        z = 0.1 * np.cos(t)
        ax.plot3D(x, y, z, color=c, linewidth=0.75)
        ax.text(0.1, 0, -0.1, '$\\theta$', 'x', horizontalalignment='center', color=c, fontsize=12)
        ax.quiver(0, 0, -0.1, -0.02, 0, 0, color=c, arrow_length_ratio=1, pivot='tip', linewidth=0.75)
    def draw_phi(ax, c='k'):
        t = np.linspace(0.1, 2*np.pi, 31)
        x = 0.1 * np.cos(t)
        y = 0.1 * np.sin(t)
        z = np.zeros(np.size(x))
        ax.plot3D(x, y, z, color=c, linewidth=0.75)
        ax.text(0.1, 0.1, 0, '$\\varphi$', 'x', horizontalalignment='center', color=c, fontsize=12)
        ax.quiver(0.1, 0, 0, 0, 0.02, 0, color=c, arrow_length_ratio=1, pivot='tip', linewidth=0.75)
    if kwargs.get('title', False):
        title = kwargs['title']
    else:
        title = ''
    if kwargs.get('axis', False):
        ax = kwargs['axis']
    else:
        raise Exception("'axis' paramater is necessary")
    ax = bloch_sphere(ax)
    draw_theta(ax)
    draw_phi(ax)
    for arg in args:
        label, color = '|ψ❭', 'r'
        if type(arg) == tuple:
            if len(arg) == 3: color = arg[2]
            label = arg[1]
            arg = arg[0]
        ax.quiver(0, 0, 0, arg.x, arg.y, arg.z, color=color, linewidth=2)
        ax.text(*repel_from_center(arg.x, arg.y, arg.z), label, 'x', \
            horizontalalignment='center', fontweight='bold', color=color)
    plt.title(title)
    return ax # handler bloch sphere

# To update GUI fields
def update_fields():
    ax_txtPA['a']['widget'].set_val(f"{q.a:.3f}")
    ax_txtPA['b']['widget'].set_val(f"{q.b:.3f}")
    for k in ax_txtPA.keys():
        ax_txtPA[k]['widget'].stop_typing()
    ax_txtSphere['theta']['widget'].set_val(f"{q.theta:.3f}")
    ax_txtSphere['varphi']['widget'].set_val(f"{q.phi:.3f}")
    for k in ax_txtSphere.keys():
        ax_txtSphere[k]['widget'].stop_typing()
    ax_txtCartesian['x']['widget'].set_val(f"{q.x:.3f}")
    ax_txtCartesian['y']['widget'].set_val(f"{q.y:.3f}")
    ax_txtCartesian['z']['widget'].set_val(f"{q.z:.3f}")
    for k in ax_txtCartesian.keys():
        ax_txtCartesian[k]['widget'].stop_typing()
    ax_qubit_orig.clear()
    ax_qubit_orig.text(0.5, 0.5, f"|$\psi$❭ = [ {q.a:.3f}, {q.b:.3f} ]$^T$ ::"
        f" Pr(0) = {q.pb_0()*100:.1f}%, Pr(1) = {q.pb_1()*100:.1f}%", 
        horizontalalignment='center', fontweight='bold', fontsize=12, color='red')
    ax_qubit_orig.set_axis_off()
    ax_gate_applied.clear()
    ax_gate_applied.set_axis_off()

# Setup GUI
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.5)
bloch_sphere_ax = plot_qubits(q, axis=ax)

# label 
ax_qubit = plt.axes((0.05, 0.9, 0.35, 0.05))
ax_qubit.text(0.5, 0.5, "Qubit", horizontalalignment='center', fontweight='bold', fontsize=14)
ax_qubit.set_axis_off()

# Ket Buttons
def fn_btnKets(event):
    for k in ax_btnKets.keys():
        if event.inaxes == ax_btnKets[k]['geom']:
            if k == '\mathcal{G}\,{\psi}':
                q.set_as(p)
            else:
                q.set_ket(k)
            break
    update_fields()
    plot_qubits(q, axis=ax)
    fig.canvas.draw() # redraw figure
ax_btnKets = dict((l, {'geom': None, 'widget': None}) \
    for l in ['0', '1', '+', '-', 'i', '-i', '\mathcal{G}\,{\psi}'])
x_ref = 0.05
y_ref = 0.85
w_ref = 0.0406
aux = ax_btnKets.keys()
for i, k in enumerate(aux):
    ax_btnKets[k]['geom'] = plt.axes((x_ref, y_ref, w_ref, 0.05)) # xpos, ypos, width, height
    ax_btnKets[k]['widget'] = Button(ax_btnKets[k]['geom'], label=f'|${k}$⟩', 
        color='blue' if i == len(aux)-1 else 'gray', hovercolor='lightblue')
    ax_btnKets[k]['widget'].on_clicked(fn_btnKets)
    x_ref += 0.0141 + w_ref

# Support
# text inputs
def eval_input(txt):
    ans = txt
    if ans == '': ans = 0
    else: ans = eval(ans.replace('pi', 'np.pi').replace('sqrt', 'np.sqrt'))
    return ans
# labels Probabilities (see update_field)
ax_qubit_orig = plt.axes((0.05, 0.5, 0.35, 0.05))
ax_gate_applied = plt.axes((0.05, 0.15, 0.35, 0.05))

# State Buttons
def fn_btnPA(event):
    a = eval_input(ax_txtPA['a']['widget'].text)
    b = eval_input(ax_txtPA['b']['widget'].text)
    q.set(a, b)
    update_fields()
    plot_qubits(q, axis=ax)
    fig.canvas.draw() # redraw figure
def fn_clearPA(event):
    for k in ax_txtPA.keys():
        if event.inaxes == ax_txtPA[k]['geo_clear']:
            ax_txtPA[k]['widget'].set_val("")
            # ax_txtPA[k]['widget'].set_active(True)
            break
ax_txtPA = dict((l, {'geom': None, 'widget': None, 'geo_clear': None, 'btn_clear': None}) for l in ['a', 'b'])
x_ref = 0.05
y_ref = 0.75
w_ref = 0.06
for k in ax_txtPA.keys():
    ax_txtPA[k]['geom'] = plt.axes((x_ref, y_ref, w_ref, 0.05)) # xpos, ypos, width, height
    ax_txtPA[k]['widget'] = TextBox(ax_txtPA[k]['geom'], label=f'{k}', color='white', hovercolor='lightgray', label_pad=0.1)
    ax_txtPA[k]['geo_clear'] = plt.axes((x_ref+w_ref, y_ref, 0.01, 0.05))
    ax_txtPA[k]['btn_clear'] = Button(ax_txtPA[k]['geo_clear'], label=f'⌫', color='gray', hovercolor='lightblue')
    ax_txtPA[k]['btn_clear'].on_clicked(fn_clearPA)
    x_ref += 0.05 + w_ref
ax_btnPA = plt.axes((x_ref, y_ref, 2.5*w_ref, 0.05)) # xpos, ypos, width, height
btnPA = Button(ax_btnPA, label=f'Set State\n$|a|^2 + |b|^2 = 1$', color='gray', hovercolor='lightblue')
btnPA.on_clicked(fn_btnPA)

# Spherical coord Buttons
def fn_btnSphere(event):
    theta = eval_input(ax_txtSphere['theta']['widget'].text)
    phi = eval_input(ax_txtSphere['varphi']['widget'].text)
    q.set_spherical(theta, phi)
    update_fields()
    plot_qubits(q, axis=ax)
    fig.canvas.draw() # redraw figure
def fn_clearSphere(event):
    for k in ax_txtSphere.keys():
        if event.inaxes == ax_txtSphere[k]['geo_clear']:
            ax_txtSphere[k]['widget'].set_val("")
            # ax_txtSphere[k]['widget'].set_active(True)
            break
ax_txtSphere = dict((l, {'geom': None, 'widget': None, 'geo_clear': None, 'btn_clear': None}) for l in ['theta', 'varphi'])
x_ref = 0.05
y_ref = 0.65
w_ref = 0.06
for k in ax_txtSphere.keys():
    ax_txtSphere[k]['geom'] = plt.axes((x_ref, y_ref, w_ref, 0.05)) # xpos, ypos, width, height
    ax_txtSphere[k]['widget'] = TextBox(ax_txtSphere[k]['geom'], label=f'$\{k}$', color='white', hovercolor='lightgray', label_pad=0.1)
    ax_txtSphere[k]['geo_clear'] = plt.axes((x_ref+w_ref, y_ref, 0.01, 0.05))
    ax_txtSphere[k]['btn_clear'] = Button(ax_txtSphere[k]['geo_clear'], label=f'⌫', color='gray', hovercolor='lightblue')
    ax_txtSphere[k]['btn_clear'].on_clicked(fn_clearSphere)
    x_ref += 0.05 + w_ref
ax_btnSphere = plt.axes((x_ref, y_ref, 2.5*w_ref, 0.05)) # xpos, ypos, width, height
btnSphere = Button(ax_btnSphere, label=f'Set Spherical (unit sphere)\n$0\leq\\theta\leq\pi;\;0\leq\\varphi<2\pi$', color='gray', hovercolor='lightblue')
btnSphere.on_clicked(fn_btnSphere)

# Cartesian coord Buttons
def fn_btnCartesian(event):
    x = eval_input(ax_txtCartesian['x']['widget'].text)
    y = eval_input(ax_txtCartesian['y']['widget'].text)
    z = eval_input(ax_txtCartesian['z']['widget'].text)
    q.set_cartesian(x, y, z)
    update_fields()
    plot_qubits(q, axis=ax)
    fig.canvas.draw() # redraw figure
def fn_clearCartesian(event):
    for k in ax_txtCartesian.keys():
        if event.inaxes == ax_txtCartesian[k]['geo_clear']:
            ax_txtCartesian[k]['widget'].set_val("")
            break
ax_txtCartesian = dict((l, {'geom': None, 'widget': None, 'geo_clear': None, 'btn_clear': None}) \
    for l in ['x', 'y', 'z'])
x_ref = 0.05
y_ref = 0.55
w_ref = 0.04
for k in ax_txtCartesian.keys():
    ax_txtCartesian[k]['geom'] = plt.axes((x_ref, y_ref, w_ref, 0.05)) # xpos, ypos, width, height
    ax_txtCartesian[k]['widget'] = TextBox(ax_txtCartesian[k]['geom'], label=f'{k}', color='white', 
        hovercolor='lightgray', label_pad=0.1) 
    ax_txtCartesian[k]['geo_clear'] = plt.axes((x_ref+w_ref, y_ref, 0.01, 0.05))
    ax_txtCartesian[k]['btn_clear'] = Button(ax_txtCartesian[k]['geo_clear'], label=f'⌫', color='gray', 
        hovercolor='lightblue')
    ax_txtCartesian[k]['btn_clear'].on_clicked(fn_clearCartesian)
    x_ref += 0.05 + w_ref
ax_btnCartesian = plt.axes((x_ref, y_ref, 2.5*w_ref, 0.05)) # xpos, ypos, width, height
btnCartesian = Button(ax_btnCartesian, label=f'Set Cartesian\n$x^2 + y^2 + z^2 = 1$', color='gray', 
    hovercolor='lightblue')
btnCartesian.on_clicked(fn_btnCartesian)

# label 
ax_quantum = plt.axes((0.05, 0.45, 0.35, 0.05))
ax_quantum.text(0.5, 0.5, "Quantum Gates (1 qubit)", horizontalalignment='center', fontweight='bold', fontsize=14)
ax_quantum.set_axis_off()

# Gate Buttons
def fn_btnGates(event):
    for k in ax_btnGates.keys():
        if event.inaxes == ax_btnGates[k]['geom']:
            h = k[1]
            if 'phi' in k[0]:
                phi = sldPhiGate.val * np.pi
                eval(f"p.set_as(q.apply_{k[0]}({phi}, to_self=False))")
            else:
                eval(f"p.set_as(q.apply_{k[0]}(to_self=False))")
            break
    update_fields()
    plot_qubits(q, (p, f"{h}|ψ❭", 'b'), axis=ax)
    if 'phi' in h:
        s = f"{h}" + "$_{=" + f"{sldPhiGate.val:.3f}" + "\pi}$" + \
            f"|$\psi$❭ = [ {p.a:.3f}, {p.b:.3f} ]$^T$ :: Pr(0) = {p.pb_0()*100:.1f}%, Pr(1) = {p.pb_1()*100:.1f}%"
    else:
        s = f"{h} |$\psi$❭ = [ {p.a:.3f}, {p.b:.3f} ]$^T$ :: Pr(0) = {p.pb_0()*100:.1f}%, Pr(1) = {p.pb_1()*100:.1f}%"
    ax_gate_applied.text(0.5, 0.5, s, horizontalalignment='center', fontweight='bold', fontsize=12, color='blue')
    fig.canvas.draw() # redraw figure
ax_btnGates = dict((l, {'geom': None, 'widget': None}) \
    for l in [
        ('X', '$X$'),
        ('Y', '$Y$'),
        ('Z', '$Z$'),
        ('ID', '$ID$'),
        ('H', '$H$'),
        ('sqNOT', '$\sqrt{NOT}$'),
        ('S', '$S$'),
        ('S_cross', '$S^\dagger$'),
        ('T', '$T$'),
        ('T_cross', '$T^\dagger$'),
        ('Rx_phi', '$R^x_\phi$'),
        ('Ry_phi', '$R^y_\phi$'),
        ('Rz_phi', '$R^z_\phi$'),
    ])
y_ref = 0.5
for i, k in enumerate(ax_btnGates.keys()):
    if i % 6 == 0: 
        x_ref = 0.05
        w_ref = 0.05
        y_ref -= 0.1
    ax_btnGates[k]['geom'] = plt.axes((x_ref, y_ref, w_ref, 0.05)) # xpos, ypos, width, height
    ax_btnGates[k]['widget'] = Button(ax_btnGates[k]['geom'], label=k[1], 
        color='gray', hovercolor='lightblue')
    ax_btnGates[k]['widget'].on_clicked(fn_btnGates)
    x_ref += 0.0141 + w_ref
ax_sldPhiGate = plt.axes((x_ref+0.03, y_ref+0.0125, 0.25, 0.025)) # xpos, ypos, width, height 
sldPhiGate = Slider(ax_sldPhiGate, '$\phi \cdot 1/\pi$', valmin=0., valmax=2., valinit=0.5, valstep=1/12)
def update(val):
    fig.canvas.draw() # redraw figure
sldPhiGate.on_changed(update)

# Reset View 3D Graph
def fn_rstGraph(event):
    bloch_sphere_ax.view_init(elev=30, azim=-60)
ax_rstGraph = plt.axes((0.05, 0.05, 0.37, 0.05)) # xpos, ypos, width, height
btn_rstGraph = Button(ax_rstGraph, label='Reset View', 
        color='gray', hovercolor='lightblue')
btn_rstGraph.on_clicked(fn_rstGraph)

# Starting app
update_fields()
plt.show()