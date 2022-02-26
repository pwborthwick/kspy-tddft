from __future__ import division
import numpy as np
from ks_util import out
from ks_aello import aello


class LR_TDA(object):
    #Time-Dependent DFT in Tamm-Dancoff Approximation

    def __init__(self, lr_data, roots=5, excitation='singlet'):

        self.c, self.eo, self.ao, self.weights, self.eri, self.functional, self.nele, self.nocc, \
        self.mol, self.orb = lr_data.values()

        self.roots = roots
        self.nvir = None ; self.excitation = excitation

        #compute the response by solving Casida 
        self.response = self.compute_response()

    def molecular_spin_rho(self):
        #transform from ao to spin mo

        ngrids, nao = self.ao.shape
        #occupations
        mo_occ = np.array([1*(i < self.nocc) for i in range(nao)]) 
        is_occupied = mo_occ > 1e-6

        #ao->mo and normalize
        c_occ = np.einsum('ij,j->ij', self.c[:,is_occupied], np.sqrt(mo_occ[is_occupied]))
        c_mo_spin = np.einsum('rp,qr->qp', c_occ, self.ao, optimize=True)

        rho = np.einsum('pi,pi->p', c_mo_spin, c_mo_spin)  

        return (rho, rho)

    def orbital_hessian(self, fxc):
        #compute orbital Hessian (A) for Tamm-Dancoff

        #orbital energies
        fo = np.diag(self.eo[:self.nocc]) ; fv = np.diag(self.eo[self.nocc:])

        #occupied to virtual excitations
        excitations = (fv.diagonal().reshape(-1,1) - fo.diagonal())
        excitations = np.ravel(excitations.transpose())

        #set required states to 'all'
        nstates = excitations.size

        #occupation mo coefficients
        co = self.c[:,:self.nocc] ; cv = self.c[:, self.nocc:] ; self.nvir = len(self.eo) - self.nocc 

        #state densities
        states = np.eye(nstates).reshape(-1, self.nocc, self.nvir)
        dov = np.einsum('xov,po,qv->xpq', states*2, co, cv)
        dov[abs(dov) < 1e-15] = 0.0

        #alpha-alpha, alpha-beta and beta-beta spins
        aa, ab, bb = fxc
        
        #singlet or triplet
        level = aa + ab if self.excitation == 'singlet' else aa - ab

        #construct reponse - compare with HF CIS spin adapted treatment
        spins, nao, _ = dov.shape
        K = np.zeros((spins, nao, nao))

        for spin in range(spins):
            sym_dov = (dov[spin] + dov[spin].T)*0.5  #xc kernel
            c_mo_spin_ov = np.einsum('rp,qr->qp',sym_dov, self.ao, optimize=True) 
            rho = np.einsum('pi,pi->p', self.ao, c_mo_spin_ov, optimize=True) 
            
            weighted_ao = np.einsum('pi,p->pi', self.ao, self.weights*level*rho) 
            K[spin] = np.einsum('pi,pj->ij', weighted_ao, self.ao, optimize=True) * 0.5

            #add in coulomb j contribution
            if self.excitation == 'singlet' : 
                K[spin] += np.einsum('rs,pqrs->pq', dov[spin], self.eri, optimize=True)

            #add in orbital energy differences
            Kov = np.einsum('xpq,po,qv->xov', K, co, cv, optimize=True)
            Kov += np.einsum('xqs,sp->xqp', states, fv, optimize=True)
            Kov -= np.einsum('xpr,sp->xsr', states, fo, optimize=True)

        return Kov.reshape(spins, -1)

    def compute_response(self):

        #compute density in molecular spin basis
        rho = self.molecular_spin_rho()[0]

        #exchange-correlation energy, first and second derivatives
        exc, vxc, fxc = self.functional('RPA', rho, spin=True)

        #solve Tamm-Dancoff
        A = self.orbital_hessian(fxc)
        e, c = np.linalg.eigh(A)

        #prepare output
        coeffs = np.zeros((self.roots, self.nocc, self.nvir))
        for i in range(self.roots):
            coeffs[i] = c[:,i].reshape(self.nocc, self.nvir)*np.sqrt(0.5)

        return (e[:self.roots], coeffs)

class TDA_properties(object):
    #class to compute transition properties

    def __init__(self, TDA, lr_data):

        self.roots = TDA.response[1].shape[0]
        self.c, self.nele, self.nocc, self.mol, self.orb = [lr_data[i] for i in ['c','electrons','occupied','molecule','basis']]
        self.nvir = self.c.shape[0] - self.nocc

        self.TDA = TDA

    def get_gauge(self, mol, type='origin'):
        #generate a gauge vector

        gauge = np.array([0,0,0])
        if type == 'charge_center':
            gauge = np.array([sum(j) for j in zip(*[i.number*i.center for i in mol])])
            total_charge = sum([i.number for i in mol])
            gauge /= total_charge
        elif type == 'geometric_center':
            gauge = np.array([sum(i.center) for i in mol])
            gauge /= len(mol)

        return np.array(gauge)

    def get_constant(self, type):
        #provide constant values

        if type == 'hartree->eV': return 27.211324570273
        if type == 'wavenumber nm': return 2.194746313702e-2

    def compute_dipole(self, type, coeff):
        #get the requested transition dipole value

        #get gauge for length gauge properties
        if type in ['dipole', 'angular']:
            gauge = self.get_gauge(self.mol, 'charge_center')
        else:
            gauge = self.get_gauge(self.mol)

        #get required dipole, angular or nabla components
        dipole_components = np.array(aello(self.mol, self.orb, type, gauge=gauge))

        #shape components
        co = self.c[:,:self.nocc] ; cv = self.c[:, self.nocc:]
        rov = np.einsum('rp,xrs,sq->xpq', co, dipole_components, cv, optimize=True)

        td = np.einsum('pq,xpq->x', 2.0*coeff, rov, optimize=True) 

        os = np.sum(td*td)

        return td, os

    def transition_properties(self, silent=False):
        #compute transition properties for each root

        #list for results dictionaries
        roots_data = []

        for root in range(self.roots):

            #dictionary for computed values
            results = {}

            results['i'] = root+1
            results['t'] = self.TDA.excitation

            #response energy and eigenvector
            excitation_energy = self.TDA.response[0][root]
            excitation_vector = self.TDA.response[1][root]

            #wavenumber in nm
            wave_number = 1/(excitation_energy * self.get_constant('wavenumber nm'))
            results['w'] = wave_number

            #details of the primary excitation 
            principal_jump  = np.unravel_index(np.argmax(np.abs(excitation_vector.ravel())), (self.nocc,self.nvir))
            principal_value = self.TDA.response[1][root][principal_jump]
            principal_jump = list(principal_jump)
            principal_jump[1] += self.nocc

            results['e'] = excitation_energy*self.get_constant('hartree->eV') 
            results['j'] = principal_jump ; results['v'] = principal_value

            #electric dipole in length gauge
            results['d'], os = self.compute_dipole('dipole', excitation_vector)
            oscillator_strength =  2 /3 * os * excitation_energy
            results['od'] = oscillator_strength ; results['sd'] = os

            #get electric dipole in velocity gauge
            results['n'], os = self.compute_dipole('nabla', excitation_vector)
            oscillator_strength =  2 /(3 * excitation_energy) * os
            results['on'] = oscillator_strength ; results['sn'] = os


            #magnetic dipole in length gauge
            results['a'], os = self.compute_dipole('angular', excitation_vector)

            #rotary strengths
            results['rl'] = sum(results['d'] * results['a'])     
            results['rv'] = -sum(results['n'] * results['a']) / excitation_energy    

            if not silent: out(results, 'TDA')

            roots_data.append(results)

        return roots_data

    def transition_NO(self, root=0, threshold=0.5):
        #Transition natural orbitals

        if root >= self.roots:
            print('only', self.roots, 'roots are available')
            return None, None

        #get coefficient for requested root
        excitation_vector = self.TDA.response[1][root]
        excitation_vector /= np.linalg.norm(excitation_vector)
        excitation_vector[abs(excitation_vector)<1e-15] = 0

        #do singular-value decomposition of transition coefficient
        u, s, v = np.linalg.svd(excitation_vector)
        v = v.T

        #get index of row which absolute maximum element for each column
        row_index_u = np.argmax(abs(u), axis=0) ; row_index_v = np.argmax(abs(v), axis=0)
        #get value of maximum element for each row
        row_max_u = u[row_index_u, np.arange(self.nocc)] ; row_max_v = v[row_index_v, np.arange(self.nvir)] 
        #if maximum element is negative swap sign of column
        u[:,row_max_u < 0] *= -1  ; v[:,row_max_v < 0] *= -1  
        
        #the combined traces should equal number of doubly occupied orbitals
        assert np.trace(u)+np.trace(v), self.nele//2

        #shape components
        co = self.c[:,:self.nocc] ; cv = self.c[:, self.nocc:]
        NTOocc = np.einsum('pi,ij->pj', co, u, optimize=True)
        NTOvir = np.einsum('pa,ab->pb', cv, v, optimize=True)

        #principal NOs
        no = np.where(abs(u[:,0]) > threshold)[0] ; threshold_occupied = [no[0], u[no,0][0]]
        no = np.where(abs(v[:,0]) > threshold)[0] ; threshold_virtual  = [no[0]+self.nocc, v[no,0][0]]

        out([root, self.TDA.response[0][root]*self.get_constant('hartree->eV'), (s*s)[0], threshold_occupied, threshold_virtual], 'NOS')

        return  np.hstack((NTOocc, NTOvir)), s*s

    def spectrum(self, data, roots, excitation='singlet', type='length'):
        #plot oscillator spectrum

        if roots > self.c.shape[0]+1: return

        def lorentzian(e0, e, tau):
        #Lorentzian broadening

           gamma = 1.0/tau
           g = (gamma/2.0)**2.0/((e0-e)**2.0 + (gamma/2.0)**2.0)

           return g

        #get the properties for requested number and type
        tda = LR_TDA(data, roots, excitation)
        properties = TDA_properties(tda, data)
        property_dictionary = properties.transition_properties(silent=True)

        if type ==  'length':
            oscillator = [property_dictionary[i]['od'] for i in range(roots)]
        elif type == 'velocity':
            oscillator = [property_dictionary[i]['on'] for i in range(roots)]

        title = '(' + type + ')'
        oscillator /= np.linalg.norm(oscillator)

        e = [property_dictionary[i]['e'] for i in range(roots)]
        jumps = [property_dictionary[i]['j'] for i in range(roots)]

        import matplotlib.pyplot as py

        #plot the oscillator spectrum
        py.title('oscillator spectrum ' + title)
        py.grid()   

        margin, tau, npoints = [0.5, 10, 100]

        for i in range(roots):
            start  = e[i] - margin
            finish = e[i] + margin
            x = np.linspace(start, finish, npoints)

            points = lorentzian(e[i], x, tau) * oscillator[i]

            py.plot(x , points, 'k')
            py.bar(e[i], oscillator[i], width= 0.1, color='orange')

            if oscillator[i] >= 0.2:
                py.text(e[i], oscillator[i], str(jumps[i][0]) + '->' + str(jumps[i][1]), fontsize='x-small')

        py.xlabel('excitation energy (eV)')
        py.ylabel('oscillator strength (norm)')  

        py.show()

class RT_TDA(object):
    #class for real-time TDDFT

    def __init__(self, rt_data, rt_field, rt_execute):

        s, self.f, self.eri, self.h_core, self.d, self.co, self.mol, self.orb, self.ao,  \
        self.weights, self.evaluate_rho, self.functional, self.evaluate_vxc = rt_data.values()

        self.field = rt_field
        self.nbf = s.shape[0]

        self.steps, self.tick, units, self.gauge = rt_execute.values()

        #primary observables storage
        self.observables = {'dipoles':[]}

        #time units
        self.time_unit, self.mu_unit = units
        self.time_conversion = 2.4188843265857e-17
        if self.time_unit == 'fs': self.time_conversion *= 1e-15/self.time_conversion
        if self.time_unit == 'ps': self.time_conversion *= 1e-12/self.time_conversion
        if self.time_unit == 'au': self.time_conversion = 1.0
        self.field['center'] *= self.tick

        from scipy.linalg import fractional_matrix_power as fractPow
        self.x = fractPow(s, -0.5)
        self.u = fractPow(s, 0.5)

    def magnus(self, axis):
        #do the 2nd order Magnus propogation

        for direction in axis:
            numerical_axis = ['x','y','z'].index(direction)
            time, field, dipole_component = self.magnus_component(self.d, self.f, numerical_axis)

            #build observables dictionary
            if np.any(self.observables.get('time')) == None: self.observables['time'] = time
            if np.any(self.observables.get('field')) == None: self.observables['field'] = field
            self.observables[direction] = dipole_component
            self.observables['dipoles'].append(direction)

    def magnus_component(self, d, f, axis):
        #second order Magnus expansion

        from scipy.linalg import expm


        iterations = int(self.steps)
        h = -1j * self.tick

        #compute gauge origin
        if self.gauge == 'charge center':
            gauge = np.array([sum(j) for j in zip(*[i.number*i.center for i in self.mol])])
            total_charge = sum([i.number for i in self.mol])
            gauge /= total_charge
        else:
            gauge = [0.0, 0.0, 0.0]

        #initial orthogonal matrices
        dp = np.dot(self.u, np.dot(d, self.u.T))
        fp = np.dot(self.x.T , np.dot(f , self.x))

        def pulse(t):

            #instantaneos pulse
            if self.field['shape'] == 'kick':
               if t == self.field['center']: return 1.0
               else: return 0.0
            #gaussian profile
            elif self.field['shape'] == 'gaussian':
                rho = self.field['rho']
                return np.exp(-((t - self.field['center'])**2)/ (2.0 * rho * rho))

        def update_field(t):
            #add dipole field to fock - in orthogonal basis

            external_field = pulse(t) * self.field['intensity']
            dipole = np.array(aello(self.mol, self.orb, 'dipole', gauge=gauge)[axis])

            induced_fock = external_field * dipole

            return np.einsum('pr,rs,qs->pq', self.x, induced_fock, self.x, optimize=True)



        def update_fock(d):
            #updating Fock on grid

            rho = self.evaluate_rho(d, self.ao, self.weights)
            exc, vxc, _ = self.functional('RPA', rho)

            f = self.h_core.astype('complex') + 2.0 * np.einsum('rs,pqrs->pq', d, self.eri.astype('complex'), optimize=True)

            return f + self.evaluate_vxc(vxc, self.ao, self.weights)

        def update_state(u, cycleDensity):
            #propagate time U(t) -> U(t+timeIncrement)

            dp = np.einsum('pr,rs,qs->pq', u, cycleDensity, np.conjugate(u), optimize=True)

            #build fock in non-orthogonal ao basis
            d = np.einsum('pr,rs,qs->pq', self.x, dp, self.x, optimize=True)
            f = update_fock(d)

            #orthogonalize for next step
            fp = np.einsum('rp,rs,sq->pq', self.x, f, self.x, optimize=True)

            return fp, dp

        TIME = []
        DIPOLE = []
        FIELD = []

        nbf = dp.shape[0]

        nuclear_dipole = 0.0
        for i in range(0, len(self.mol)):
            #nuclear and gauge displacement
            nuclear_dipole += self.mol[i].number * (self.mol[i].center[axis] - gauge[axis])


        for cycle in range(iterations):

            k = np.zeros((2,nbf,nbf)).astype('complex')

            mu = -2.0 * np.trace(np.dot(d, aello(self.mol, self.orb, 'dipole', gauge=gauge)[axis])) + nuclear_dipole

            DIPOLE.append(mu.real)
            TIME.append(cycle * self.tick)
            FIELD.append(pulse(cycle * self.tick)*self.field['intensity'])

            cycle_density = dp.copy()

            #propogation setep 1
            k[0] = h * (fp + update_field(cycle * self.tick))
            propogation = expm(k[0])
            fp, dp = update_state(propogation, cycle_density)

            #propogation setep 2
            k[1] = h * (fp + update_field((cycle+1) * self.tick))
            propogation = expm(0.5*(k[0] + k[1]))
            fp, dp = update_state(propogation, cycle_density)

            #unorthogonalise for energy calculation
            d = np.einsum('pr,rs,qs->pq', self.x, dp, self.x, optimize=True)
            
        return np.array(TIME), np.array(FIELD), np.array(DIPOLE)
        
    def display(self, axis):
        #plot the observable

        units = {'au->debye':2.541580253,  'hartree->eV' : 27.21138505}
        import matplotlib.pyplot as plt


        fig, ax_dipole = plt.subplots(3, sharex=True, figsize=(5,4))
        fig.suptitle(' Magnus Dipole Propogation', fontsize='small')
        for i in range(3):
            axis = ['x','y','z'][i]
            plt.grid()
            ax_dipole[i].tick_params(axis='both', labelsize='x-small')
            ax_dipole[i].set_xlabel('time (' + self.time_unit + ')',fontsize='x-small')
            ax_dipole[i].set_ylabel(axis + '-dipole (' + self.mu_unit + ')', fontsize='x-small')
            ax_dipole[i].plot(self.observables['time'], self.observables[axis] * units['au->debye'], 'k', linewidth=0.4)

            ax_field = ax_dipole[i].twinx()
            ax_field.tick_params(axis='both', labelsize='x-small')
            ax_field.set_ylabel('field (eV)', fontsize='x-small')
            ax_field.plot(self.observables['time'], self.observables['field'] * units['hartree->eV'], 'orange', linewidth=0.2)

        fig.tight_layout()
        plt.show()

    def get_observables(self):
        return self.observables

    def get_peaks(self, amplitudes, frequency, tolerance):
        #find the peaks in the spectrum aove tolerance

        from scipy.signal import argrelmax as pks

        maxima = pks(abs(amplitudes))

        #apply tolerance
        idx = np.where((abs(amplitudes[maxima]) >= tolerance))
        jdx = maxima[0][idx[0]] 

        peak_count = len(jdx)
        peaks = np.zeros(peak_count)
        for i in range(peak_count):
            peaks[i] = frequency[jdx][i]

        return peaks


    def spectrum(self, data):
        #plot a frequency domain spectra using Pade approximants

        from scipy.linalg import toeplitz
        import matplotlib.pyplot as plt

        #fine-structure constant and conversions
        units = {'alpha':0.00729735256,  'hartree->eV' : 27.21138505}

        #prepare frequency - x-axis
        f_range = np.arange(0, data['eV range'], data['resolution'])
        f = np.exp(-1j * f_range * self.tick)

        #storage for results
        SPECTRA = {'frequency':f_range}
        PEAKS   = {}

        norm_factor = 0

        #loop over available dipole components
        for axis in self.observables['dipoles']:

            dipole = self.observables[axis]
            dipole -= dipole[0]

            #convolute damping to dipole - Lorentian type
            steps = len(dipole)
            dipole *= np.exp(-(self.tick * np.arange(steps))/data['damping'])

            #generate vector for solve equation - symmetry redundant
            n = steps//2
            X = -dipole[n+1 : 2*n]

            #generate matrix for solve equation
            A = toeplitz(dipole)[n:2*n-1, :n-1]

            #solve system - then column vector with leading 1 - dimension n+1
            b = np.linalg.solve(A, X)
            b = np.hstack((1,b))

            a =  np.einsum('ij,j->i',np.tril(toeplitz(dipole[:n])), b, optimize=True)

            #extended Euclidean algorithm
            f_weight = np.poly1d(a)(f) / np.poly1d(b)(f)

            omega = data['function'](f_weight)
            f_amplitude = (4.0*np.pi*units['alpha']*f_range*(omega))/self.field['intensity']

            SPECTRA[axis] = f_amplitude
            norm_factor = max(norm_factor, np.max(SPECTRA[axis]))

            PEAKS[axis] = self.get_peaks(f_amplitude, f_range, 0.1) * units['hartree->eV']

        colors = {'x':'orange', 'y':'k', 'z':'violet'}
        for axis in self.observables['dipoles']:

            plt.plot(SPECTRA['frequency']*units['hartree->eV'], SPECTRA[axis]/norm_factor, label=axis, color=colors[axis])
        
        plt.title('Magnus Dipole Propogation')
        plt.legend(loc=1)
        plt.xlabel('Energy (eV)')
        plt.ylabel('scaled $\sigma(\omega)$ [arb. units]')
        plt.grid()
        plt.show()

        #print peaks in eV
        print('RT-TDA dipole transition energies (eV)\n--------------------------------------')
        for axis in self.observables['dipoles']:
            print('axis-',axis,'      ', end='')
            for i in PEAKS[axis]:
                print('{:>5.2f}  '.format(i), end='')
            print()

        return PEAKS 