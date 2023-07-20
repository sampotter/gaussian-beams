import colorcet as cc
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg

from pathlib import Path
from matplotlib.colors import LogNorm
from scipy.special import j0, j1
from scipy.special import hankel1 as h1

j2 = lambda z: scipy.special.jn(2, z)

from util import *

nlam = 80 # = diam*k/2pi
d = 2 # R^d
k = 2*np.pi*nlam/np.sqrt(2)
N = int(np.ceil(3*k))
h = 1/N # Grid spacing
L = 1.5 # A little bigger than sqrt(d)

print(f'{nlam = }')
print(f'{k = }')
print(f'{N = }')

@np.vectorize
def get_GkL(s, L, k):
    denom = s**2 - k**2
    Lk = L*k
    if denom == 0:
        return 1j*L*np.pi*((2*j1(Lk) - Lk*j2(Lk))*h1(0, Lk) + k*L*j1(Lk)*h1(1, Lk))/(4*k)
    else:
        Ls = L*s
        return (1 + (1j*np.pi/2)*(Ls*j1(Ls)*h1(0, Lk) - Lk*j0(Ls)*h1(1, Lk)))/denom

def fft(F):
    return np.fft.fft2(np.fft.ifftshift(F)*h**2)

def ifft(FHat):
    return np.fft.fftshift(np.fft.ifft2(FHat, norm='forward'))

# Constrast function:
q = lambda x, y: -np.exp(-((4*np.sqrt(x**2 + y**2))**8)/2)

# Incident field:
phi_in = lambda x, y: np.exp(1j*k*x)

# Set up frequency grid (I and J) and space grid (X and Y):
I, J = np.meshgrid(np.arange(-N//2, N//2), np.arange(-N//2, N//2))
shape = I.shape
m, n = shape
X, Y = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n))
R = np.sqrt(X**2 + Y**2)
S = np.sqrt(I**2 + J**2)
xy_extent = [X.min(), X.max(), Y.min(), Y.max()]

Q = q(X, Y)
PhiIn = phi_in(X, Y)
Rhs = -k**2*Q*PhiIn
GkLHat = np.fft.fftshift(get_GkL(np.pi*S, L/2, k))

def V(F):
    return ifft(GkLHat*fft(F))

num_iter = 0

def K_mul(sigma):
    # NOTE: this isn't actually sigma from inside bicgstab!
    global num_iter
    assert np.isfinite(sigma).all()
    num_iter += 1
    print(num_iter)
    sigma = sigma.reshape(shape)
    return (sigma + k**2*Q*V(sigma)).ravel()

frame = 0

def make_Phi_plot_from_volume_density(sigma, title=None, vmin=None, vmax=None, **kwargs):
    Phi = PhiIn + V(sigma)
    if vmin == 'negabsmax':
        vmin = -abs(Phi).max()
    if vmax == 'absmax':
        vmax = abs(Phi).max()
    plt.figure(figsize=(10, 8))
    plt.imshow(np.real(Phi), extent=[-1, 1, -1, 1], cmap=cc.cm.gouldian, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlim(-1/2, 1/2)
    plt.ylim(-1/2, 1/2)
    plt.gca().set_aspect('equal')
    T = np.linspace(0, 2*np.pi)
    r = 1/4
    plt.plot(r*np.cos(T), r*np.sin(T), c='k', linewidth=1, linestyle='--')
    if title is not None:
        plt.title(title)
    plt.tight_layout()

def callback(sigma):
    global num_iter, frame
    sigma = sigma.reshape(shape)
    make_Phi_plot_from_volume_density(sigma, f'iteration {num_iter}', vmin=-7, vmax=7)
    path = Path('frames')/f'{frame:04}.png'
    plt.savefig(path)
    plt.close()
    frame += 1

N = np.prod(shape)
KMulOp = scipy.sparse.linalg.LinearOperator((N, N), K_mul, dtype=np.complex128)

# sigma, code = scipy.sparse.linalg.gmres(KMulOp, Rhs.ravel(), x0=None, tol=1e-15, restart=2*k, maxiter=1)
sigma, code = scipy.sparse.linalg.bicgstab(KMulOp, Rhs.ravel(), x0=Rhs.ravel(), tol=1e-15, callback=callback)
sigma = sigma.reshape(shape)

print(f'{num_iter = }, {code = }')

make_Phi_plot_from_volume_density(sigma)
plt.show()
