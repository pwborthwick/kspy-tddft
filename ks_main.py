from __future__ import division
from ks_grid import GRID
from ks_util import dii_subspace, out
import numpy as np

units = {'bohr->angstrom' : 0.52917721092}
elements = {'H':1, 'O':8}

class atom(object):
    #molecule class

    def __init__(self, symbol, center):

        self.symbol = symbol
        self.number = elements[symbol]
        self.center = np.array(center)

class orbital(object):
    #basis class

    def __init__(self, atom, momentum, exponents, coefficients, normalization, atoms):

        self.atom     = atom
        self.momentum = np.array(momentum)
        self.ex       = np.array(exponents)
        self.co       = np.array(coefficients)
        self.normal   = np.array(normalization)
        self.center   = atoms[atom].center

def nuclear_repulsion(mol):
    #nuclear repulsion

    eNuc = 0.0

    atoms = len(mol)
    for i in range(atoms):
        for j in range(i+1, atoms):
            r = np.linalg.norm(mol[i].center-mol[j].center)
            eNuc += mol[i].number*mol[j].number / r

    return eNuc

def lda_functional(name, rho, spin=False):
    #exchange-correlation functional VWN3

    if name == 'RPA':

        epsilon = 1.0e-30
        rho = rho + epsilon  


        if not spin:
            #Local Density Approximation - unpolarized
            ex_lda = -(3.0/4.0) * pow(3.0/np.pi,1.0/3.0) * pow(rho,1.0/3.0)
      
            vx_lda = 4/3 * ex_lda

            #Random Phase Approximation - unpolarized
            p = pow(3/(4*np.pi*rho),1/3)
            ec_rpa = 0.0311*np.log(p) - 0.048 + 0.009*p*np.log(p) - 0.017*p
     
            rpa = -311/30000 - 0.003*np.log(p)*p + p/375
            vc_rpa = ec_rpa + rpa

            exc = ex_lda + ec_rpa
            vxc = vx_lda + vc_rpa
            fxc = None

        else:
            #Local Density Approximation - polarized
            ex_lda = -(3.0/4.0) * pow(3.0/np.pi,1.0/3.0) * pow(rho,1.0/3.0) * pow(2.0,1.0/3.0)
            vx_lda = 4/3 * ex_lda
            fx_lda = 4/9 * ex_lda / rho

            #Random Phase Approximation - polarized
            p = pow(3/ (8*np.pi*rho),1/3)
            ec_rpa = 0.0311*np.log(p) - 0.048 + 0.009*p*np.log(p) - 0.017*p

            rpa = -311/30000 - 0.003*np.log(p)*p + p/375
            vc_rpa = ec_rpa +  rpa
           
            fc_rpa = (-311/60000  - 0.001 * p * np.log(p) + 1/720 * p)/rho

            exc = ex_lda + ec_rpa
            vxc = vx_lda + vc_rpa
            fxc = fx_lda + fc_rpa

            #fx_lda[1] = 0
            fxc = (fxc, fc_rpa, fxc)


        exc[abs(exc) > 1e9] = 0.0
        vxc[abs(vxc) > 1e8] = 0.0

        return exc, vxc, fxc

def evaluate_gto(gto, p):
    #compute the value of gaussian density at (x,y,z)

    A = (p - gto.center) ; L = np.prod(A**gto.momentum, axis=1).reshape(-1,1)

    phi = np.sum(L*gto.normal*gto.co*np.exp(-gto.ex*np.sum(A*A, axis=1).reshape(-1,1)), axis=1)

    return phi.reshape(-1,1)

def evaluate_atomic_orbital(basis, p):
    #evaluate the GTO of the atomic orbitals of the molecule

    ao = []
    for i in basis:
        ao.append(evaluate_gto(i, p))

    return np.hstack(ao)

def evaluate_rho_lda(d, ao, weights):
    #evaluate the density over grid shells

    d = d + d.T
    ao_density = np.einsum('pr,rq->pq', ao, d, optimize=True)
    ao_density = np.einsum('pi,pi->p', ao, ao_density, optimize=True)

    #set small values to np.zeros
    ao_density[abs(ao_density) < 1.0e-15] = 0

    return ao_density

def evaluate_vxc(vxc, ao, weights):
    #construct exchange-correlation matrix

    weighted_ao = np.einsum('pi,p->pi', ao, 0.5*weights*vxc, optimize=True)
    xc = np.einsum('rp,rq->pq', ao, weighted_ao, optimize=True)

    return xc + xc.T

def evaluate_exc(exc, rho, weights):
    #evaluate exchange-correlation energy

    return np.einsum('p,p->', rho*weights, exc, optimize=True)

if __name__ == '__main__':

    mesh = 'close' ; functional = 'RPA'; DIIS = True ; DIIS_SIZE = 6

    #define the molecule atoms first then basis (sto-3g)
    mol = []
    mol.append(atom('O', [0.0,0.0,0.0])) ; mol.append(atom('H', [0,-0.757 ,0.587])) ; mol.append(atom('H', [0,0.757,0.587]))
    for m in mol:
        m.center /= units['bohr->angstrom']

    orb = []
    orb.append(orbital(0, [0,0,0], [130.7093214, 23.80886605, 6.443608313], [0.1543289672962566, 0.5353281422870151, 0.44463454218921483],   \
                                   [27.551167822078394, 7.681819989204459, 2.882417873168662], mol))
    orb.append(orbital(0, [0,0,0], [5.033151319, 1.169596125, 0.38038896],  [-0.09996722918837482, 0.399512826093505, 0.7001154688886181],   \
                                   [2.394914882501622, 0.8015618386293724, 0.34520813393821864], mol))
    orb.append(orbital(0, [1,0,0], [5.033151319, 1.169596125, 0.38038896],  [0.15591627500155536, 0.6076837186060621, 0.39195739310391],     \
                                   [10.745832634231427, 1.7337440707285054, 0.4258189334467701], mol))
    orb.append(orbital(0, [0,1,0], [5.033151319, 1.169596125, 0.38038896],  [0.15591627500155536, 0.6076837186060621, 0.39195739310391],     \
                                   [10.745832634231427, 1.7337440707285054, 0.4258189334467701], mol))
    orb.append(orbital(0, [0,0,1], [5.033151319, 1.169596125, 0.38038896],  [0.15591627500155536, 0.6076837186060621, 0.39195739310391],     \
                                   [10.745832634231427, 1.7337440707285054, 0.4258189334467701], mol))
    orb.append(orbital(1, [0,0,0], [3.425250914, 0.6239137298, 0.168855404], [0.15432896729459913, 0.5353281422812658, 0.44463454218443965], \
                                   [1.7944418337900938, 0.5003264922111158, 0.1877354618463613], mol))
    orb.append(orbital(2, [0,0,0], [3.425250914, 0.6239137298, 0.168855404], [0.15432896729459913, 0.5353281422812658, 0.44463454218443965], \
                                   [1.7944418337900938, 0.5003264922111158, 0.1877354618463613], mol))

    #output details of molecule
    out([mol, DIIS, DIIS_SIZE, functional, mesh], 'initial')
    #use a reduced version of Harpy's cython integrals
    from ks_aello import aello
    s, t, v, eri = aello(mol, orb)

    #orthogonal transformation matrix
    from scipy.linalg import fractional_matrix_power as fractPow
    x = fractPow(s, -0.5)

    #inital fock is core hamiltonian
    h_core = t + v

    #orthogonal Fock
    fo = np.einsum('rp,rs,sq->pq', x, h_core, x, optimize=True )

    #eigensolve and transform back to ao basis
    eo , co = np.linalg.eigh(fo)
    c = np.einsum('pr,rq->pq', x, co, optimize=True)

    #build our initial density
    nocc = np.sum([a.number for a in mol])//2

    d = np.einsum('pi,qi->pq', c[:, :nocc], c[:, :nocc], optimize=True)

    #SCF conditions
    cycles = 50 ; tolerance = 1e-6
    out([cycles, tolerance], 'cycle')

    #get grid
    grid, weights = GRID(mol, mesh)

    #evaluate basis over grid
    ao = evaluate_atomic_orbital(orb, grid)

    last_cycle_energy = 0.0

    #diis initialisation
    if DIIS: diis = dii_subspace(DIIS_SIZE)

    #SCF loop
    for cycle in range(cycles):

        #build the coulomb integral
        j = 2.0 * np.einsum('rs,pqrs->pq', d, eri, optimize=True)

        #evalute density over mesh
        rho = evaluate_rho_lda(d, ao, weights)

        #evaluate functional over mesh
        exc, vxc, _ = lda_functional(functional, rho)

        out([cycle, np.einsum('pq,pq->', d, (2.0*h_core), optimize=True), \
                    np.einsum('pq,pq->', d, ( j), optimize=True), \
                    evaluate_exc(exc, rho, weights),np.sum(rho*weights) ],'scf')

        #evaluate potential
        vxc = evaluate_vxc(vxc, ao, weights)

        f = h_core + j + vxc

        if (cycle != 0) and DIIS:
            f = diis.build(f, d, s, x)

        #orthogonal Fock and eigen solution
        fo = np.einsum('rp,rs,sq->pq', x, f, x, optimize=True )

        #eigensolve
        eo , co = np.linalg.eigh(fo)
        c = np.einsum('pr,rq->pq', x, co, optimize=True)

        #construct new density
        d = np.einsum('pi,qi->pq', c[:, :nocc], c[:, :nocc], optimize=True)

        #electronic energy
        eSCF = np.einsum('pq,pq->', d, (2.0*h_core + j), optimize=True) + evaluate_exc(exc, rho, weights)

        if abs(eSCF - last_cycle_energy) < tolerance: break
        if DIIS: vector_norm = diis.norm
        else:    vector_norm = ''
        out([cycle, abs(eSCF - last_cycle_energy), vector_norm],'convergence')

        last_cycle_energy = eSCF

    out([eSCF, np.einsum('pq,pq->', d, (2.0*h_core), optimize=True), \
                np.einsum('pq,pq->', d, ( j), optimize=True), \
                evaluate_exc(exc, rho, weights), nuclear_repulsion(mol) ], 'final')

    out([eo, c, np.sum(rho*weights), d, s, mol, orb], 'post')


    from ks_tda import LR_TDA, TDA_properties, RT_TDA

##########
# LR-TDA #
##########
    roots = 1
    #create data object containing quantities need for TDA analysis
    tda_data = {'c':c, 'orbital energies':eo, 'atomic grid':ao, 'grid weights':weights, 'eri':eri, 'functional':lda_functional}
    tda_data.update({'electrons':int(np.sum(rho*weights)), 'occupied':nocc, 'molecule':mol, 'basis':orb})

    #create TDA object with property (energies, coefficients)
    lr_tda = LR_TDA(tda_data, roots=roots, excitation='singlet')
 
    #create TDA_properties object
    properties = TDA_properties(lr_tda, tda_data)

    #transition_properties method prints to output and returns list of results dictioaries
    property_dictionary = properties.transition_properties()

    #natural transition orbitals
    nto, sv = properties.transition_NO(root=0, threshold=0.2)

    #plot oscillator spectrum
    properties.spectrum(tda_data, 8, 'singlet', 'length' )

##########
# RT-TDA #
##########
    #prepare data for RT-TDDFT computations
    tda_data.clear()
    tda_data.update({'s':s, 'f':f, 'eri':eri, 'h_core': h_core, 'd':d, 'co':co, 'mol':mol,'orb':orb, 'ao':ao, 'weights':weights})
    tda_data.update({'evaluate_rho':evaluate_rho_lda, 'functional':lda_functional,'evaluate_vxc':evaluate_vxc})
    rt_field = {'shape': 'kick', 'intensity':0.0001, 'center':0}
    rt_execute = {'steps':200, 'tick':1, 'units':['au', 'debye'], 'gauge':'origin'}

    #instatiate RT_TDA class
    rt_tda = RT_TDA(tda_data, rt_field, rt_execute)

    #use Magnus 2nd order to do time propogation
    rt_tda.magnus(['x','y','z'])
    rt_tda.display('z')

    #spectral analysis of dipole
    peaks = rt_tda.spectrum({'damping':5000, 'eV range':1.5, 'resolution':0.0001, 'function':np.abs})
    observables = rt_tda.get_observables()
    np.savez('obs.npz', t=observables['time'], x=observables['x'], y=observables['y'], z=observables['z'])
