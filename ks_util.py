from __future__ import division
import numpy as np

class dii_subspace(object):
    #direct inversion of iterative subspace class

    def __init__(self, size):
        self.size = size
        self.fock_vector  = []
        self.error_vector = []
        self.nbf  = -1
        self.norm = 0.0

    def append(self, f, d, s, x):
        #update the subspaces respecting capacity of buffer

        self.fock_vector.append(f) 
        fds = np.einsum('im,mn,nj->ij',f, d, s, optimize=True)
        self.error_vector.append(np.einsum('mi,mn,nj->ij', x, (fds - fds.T), x, optimize=True))

        #set nbf if not set
        if self.nbf == -1: self.nbf = f.shape[0]
        self.norm = np.linalg.norm(self.error_vector[-1])

        #check capacity
        if len(self.fock_vector) > self.size:
            del self.fock_vector[0]
            del self.error_vector[0]


    def build(self, f, d, s, x):
        #compute extrapolated Fock

        #update buffers
        self.append(f, d, s, x)

        #construct B matrix
        nSubSpace = len(self.fock_vector)
        
        #start diis after cache full
        if nSubSpace < self.size: return f

        b = -np.ones((nSubSpace+1,nSubSpace+1))
        b[:-1,:-1] = 0.0 ; b[-1,-1] = 0.0
        for i in range(nSubSpace):
            for j in range(nSubSpace):
                b[i,j] = b[j,i] = np.einsum('ij,ij->',self.error_vector[i], self.error_vector[j], optimize=True)


        #solve for weights
        residuals = np.zeros(nSubSpace+1)
        residuals[-1] = -1

        try:
            weights = np.linalg.solve(b, residuals)
        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e): exit('diis failed with singular matrix')

        #weights should sum to +1
        sum = np.sum(weights[:-1])
        assert np.isclose(sum, 1.0)

        #construct extrapolated Fock
        f = np.zeros((self.nbf, self.nbf), dtype='float')
        for i in range(nSubSpace):
            f += self.fock_vector[i] * weights[i]

        return f

def out(data, key):
    #print data to console

    symbol = lambda i: ['s','px','py','pz'][[[0,0,0],[1,0,0],[0,1,0],[0,0,1]].index(i)]

    if key == 'initial':
        mol, orb = data[:2]
        units = {'bohr->angstrom' : 0.52917721092}

        print('   ks output')
        print('molecule is             water')
        bond = np.linalg.norm(mol[1].center-mol[0].center) * units['bohr->angstrom']
        a = mol[0].center - mol[1].center ; b =  mol[0].center - mol[2].center
        angle = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        print('geometry is             OH bonds - {:<4.2f}      HOH angle - {:<6.2f}'.format(bond, np.degrees(angle)))

        print('basis is                STO-3G {psi4 format}')
        print('analytic integration    aello cython - McMurchie-Davidson scheme')
        print('numerical integration   (Mura-Knowles, Lebedev)')
        print('                        radial prune is Aldrich-Treutler')
        print('                        Becke Partition scheme is Stratmann ')
        print('                        Radial adjust is Treutler')
        print('                        order: period 1 (10,11) and period 2 (15,15)')
        print('mesh                   ',data[4])
        print('functional             ',data[3])
        print('diis                   ',data[1],'  buffer size is ', data[2])

    if key == 'cycle':
        print('\nscf control             maximum cycles are ', data[0], '        convergence tolerance ', data[1])

    if key == 'scf':
        cycle, e1e, ej, ex, ne = data
        if cycle == 0:
            print('\n cycle     1 electron         coulomb         exchange          electrons')
            print('                                                                                   \u0394E         diis norm')
            print('-------------------------------------------------------------------------------------------------------')
        print('   {:>2d}     {:>12.8f}    {:>12.8f}    {:>12.8f}        {:>8.4f} '.format(cycle, e1e, ej, ex, ne))

    if key == 'convergence':
        if data[0] == 0:
            print()
        else:
            if type(data[2]) != str:
                print('                                                                             {:>12.6f}  {:>12.6f} '.format(data[1], data[2]))
            else:
                print('                                                                             {:>12.6f}            '.format(data[1]))

    if key == 'final':
        print('\nfinal energies (Hartree)\n------------------------\none electron        {:>15.10f}'.format(data[1]))
        print('coulomb             {:>15.10f}'.format(data[2]))
        print('exchange            {:>15.10f}'.format(data[3]))
        print('nuclear repulsion   {:>15.10f}'.format(data[4]))
        print('total electronic    {:>15.10f}'.format(data[0]))
        print('\nfinal total energy  {:>15.10f}'.format(data[0]+data[4]))

    if key == 'post':
        print('\nmolecular orbitals\n------------------')
        eo = data[0] ; co = data[1] ; ne = int(data[2]) ; d = data[3] ; s = data[4] ; mol = data[5] ; orb = data[6]

        mo_type = 'occupied'
        for i, e in enumerate(eo):
            print('{:>2d} {:>10.5f} {:10}'.format(i, e, mo_type))

            if (i == (ne//2)-2):    mo_type = 'homo'
            elif mo_type == 'homo': mo_type = 'lumo'
            elif mo_type == 'lumo': mo_type = 'virtual'

        print('\nmulliken populations\n--------------------')
        population = 2.0 * np.einsum('ij,ji->i', d, s, optimize=True)
        shell = [1,2,2,2,2] ; k = 0 ; atom = 0
        for i, p in enumerate(population):
            orbital = symbol(list(orb[i].momentum))
            if orb[i].atom != atom:
                atom = orb[i].atom ; k = 0
            print('{:>2d} {:>10.5f}   {:4}'.format(i, p, str(shell[k]) + orbital))
            k += 1

        print('\natomic charge\n---------------')
        charge = np.zeros(len(mol))
        for i, basis in enumerate(orb):
            charge[basis.atom] += population[i]
        for i, atom in enumerate(mol):
            charge[i] = atom.number - charge[i]
            print('{:>2d} {:>10.5f} {:4}'.format(i, charge[i], atom.symbol))

        from ks_aello import aello
        debyes = 2.541580253

        #get components of dipole matrices
        dipole_components = np.array(aello(mol, orb, 'dipole', d, [0,0,0]))

        dipole =-2* np.einsum('pi,xip->x', d, dipole_components, optimize=True)       
        for direction in range(3):

            #nuclear component and charge center adjustment
            for m in mol:
                dipole[direction] += m.number * (m.center[direction])

            dipole[direction] = dipole[direction] * debyes

        print('\ndipole momemts (Debye)\n----------------------')
        print(' x= {:<8.5f} y= {:<8.5f} z= {:<8.5f}'.format(dipole[0], dipole[1], dipole[2]))

    if key == 'TDA':
        if data['i'] == 1:
            print('\nTDA analysis for',data['t'],'states\n--------------------------------------------------------------------------')
        else:
            print('..........................................................................')

        print('root {:<2d}    energy {:<8.4f}eV   {:>6.2f} nm  f = {:<8.4f}'.format(data['i'], data['e'], data['w'], data['od']))
        jump = str(data['j'][0]) + '->' + str(data['j'][1])
        print('principal excitation {:<8s}  magnitude {:>8.4f} '.format(jump, data['v']))

        print('transition electric dipole (length gauge)(au)   {:>8.4f} {:>8.4f} {:>8.4f}'.format(data['d'][0], data['d'][1], data['d'][2]))
        print('                           norm\u00B2 {:<8.4f} oscillator strength {:<8.4f}'.format(data['sd'], data['od']))

        print('\ntransition electric dipole (velocity gauge)(au) {:>8.4f} {:>8.4f} {:>8.4f}'.format(data['n'][0], data['n'][1], data['n'][2]))
        print('                           norm\u00B2 {:<8.4f} oscillator strength {:<8.4f}'.format(data['sn'], data['on']))

        print('\ntransition magnetic dipole (length gauge)(au)   {:>8.4f} {:>8.4f} {:>8.4f}'.format(data['a'][0], data['a'][1], data['a'][2]))

        print('\nrotary strengths           (length) {:<8.4f}    (velocity) {:<8.4f}'.format(data['rl'], data['rv']))

    if key == 'NOS':
        print('\nTransition Natural Orbitals\n---------------------------')

        print('Natural transition orbitals for state {:<2d}  {:<8.4f}eV  maximum component is {:<8.4f}'.format(data[0]+1, data[1], data[2]))

        print('Inferred transition ')
        print('{:<2d}->{:>2d}   ({:<6.4f} -> {:<6.4f})'.format(data[3][0], data[4][0], data[3][1], data[4][1]))
