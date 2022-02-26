from __future__ import division
#Treutler-Ahlrich Prune scheme JCP 102, 346 (1995); DOI:10.1063/1.469408
#Stratmann Becke scheme Stratmann, Scuseria, Frisch. CPL, 257, 213 (1996)
#Mura-Knowles radial grid JCP 104, 9848 (1996); DOI:10.1063/1.471749
#Angular grid V.I. Lebedev, and D.N. Laikov, Doklady Mathematics, Vol. 59, No. 3, (1999)

import numpy as np

units = {'bohr->angstrom' : 0.52917721092}

def atom_separations(atoms):
    #distance array

    n_atoms = len(atoms)
    separations = np.zeros((n_atoms, n_atoms))
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            separations[i,j] = np.linalg.norm(a.center - b.center)

    return separations

period_table = {'H':1,'He':1,'Li':2,'Be':2,'B':2,'C':2,'N':2,'O':2,'F':2}
bragg = {'H':0.35,'He':1.40,'Li':1.45,'Be':1.05,'B':0.85,'C':0.70,'N':0.65,'O':0.60,'F':0.50, 'Ar':1.50} 


def getGridMetrics(grid_type='coarse'):
    #return the radial and angular points count

    if grid_type == 'coarse':
        radial  = {1:10, 2:15}
        angular = {1:11, 2:15}
        lebedev = {11:50, 15:86}
    elif grid_type == 'close':
        radial  = {1:50, 2:75}
        angular = {1:29, 2:29}
        lebedev = {29: 302}
    else:
        exit('grid type [', grid_type, ' ] not implemented')

    return radial, angular, lebedev

def muraKnowlesRadialGrid(species, r_points):
    #Implement the Mura-Knowles radial grid

    f = 5.2
    if species in ['Li','Be']: f = 7.0

    r = np.empty(r_points) ; dr = np.empty(r_points)

    #compute radial distances and weights
    for i in range(r_points):
        x    = (i + 0.5)/r_points
        r[i] = -f * np.log(1.0 - x*x*x)
        dr[i] = f * 3.0*x*x/((1.0 - x*x*x) * r_points)

    return r, dr

def treutlerRadialPruning(lebedev_points, r_points):
    #Teutler pruning 

    p_grid = np.empty(r_points, dtype=int)

    p_grid[:r_points//3] = 14
    p_grid[r_points//3:r_points//2] = 50
    p_grid[r_points//2:] = lebedev_points

    return p_grid

def stratmann(coordinates):
    #stratmann Becke scheme

    a = 0.64 

    m = coordinates/a
    mm = m * m
    g = np.array((1/16.0)*(m*(35.0 + mm*(-35.0 + mm*(21.0 - 5.0 *mm)))))

    g[coordinates <= -a] = -1
    g[coordinates >=  a] =  1

    return g

def treutlerAdjust(mol, radii):
    #Aldrich-Treutler adjustment using BRAGG radii

    #get Bragg radii
    atom_symbols = [a.symbol for a in mol]
    if radii == 'BRAGG':
        r = np.sqrt([bragg[s] for s in atom_symbols])
    else:
        exit('BRAGG radii only implemented')

    atoms = len(mol)
    a = np.zeros((atoms, atoms))
    for i in range(atoms):
        for j in range(i+1, atoms):
            a[i,j] = 0.25 * (r[j]/r[i] - r[i]/r[j])
            a[j,i] = -a[i,j]

    a[a < -0.5] = -0.5 ; a[a > 0.5] = 0.5
 
    return a

def ohSymmetry(points, a, b, v):
    #make spherically symmetric shell points

    if points == 0:
        n = 6 
        a = 1.0
        shell = {'-a':[4,13,22],'+a':[0,9,18],'-b':[],'+b':[],'v':[3,7,11,15,19,23]}

    elif points == 1:
        n = 12
        a = np.sqrt(0.5)
        shell = {'-a':[5,10,13,14,20,26,28,30,36,41,44,45],'+a':[1,2,6,9,16,18,22,24,32,33,37,40],'-b':[],'+b':[], \
                  'v':list(range(3, 48, 4))}

    elif points == 2:
        n = 8
        a = np.sqrt(1.0/3.0)
        shell = {'-a':[4,9,12,13,18,20,22,25,26,28,29,30],'+a':[0,1,2,5,6,8,10,14,16,17,21,24], '-b':[],'+b':[], \
                  'v':list(range(3, 32, 4))}

    elif points == 3:
        n = 24
        b = np.sqrt(1.0 - 2.0*a*a)
        shell = {'+a':[0,1,5,8,16,17,21,24,32,34,38,40,42,46,48,56,65,66,69,70,74,78,81,85], \
                 '-a':[4,9,12,13,20,25,28,29,36,44,50,52,54,58,60,62,73,77,82,86,89,90,93,94], \
                 '+b':[2,6,10,14,33,37,49,53,64,72,80,88], \
                 '-b':[18,22,26,30,41,45,57,61,68,76,84,92], \
                  'v':list(range(3, 96, 4))} 

    elif points == 4:
        n = 24
        b = np.sqrt(1.0 - a*a)
        shell = {'+a':[0,8,17,21,32,40,50,54,65,73,82,86], \
                 '-a':[4,12,25,29,36,44,58,62,69,77,90,94], \
                 '+b':[1,5,16,24,34,38,48,56,66,70,81,89], \
                 '-b':[9,13,20,28,42,46,52,60,74,78,85,93], \
                  'v':list(range(3, 96, 4))}

    elif points == 5:
        n = 48
        c = np.sqrt(1.0 - a*a - b*b)
        shell = {'+a':[0,8,16,24,32,40,48,56,65,69,81,85,98,102,106,110,129,133,145,149,162,166,170,174], \
                 '-a':[4,12,20,28,36,44,52,60,73,77,89,93,114,118,122,126,137,141,153,157,178,182,186,190], \
                 '+b':[1,5,17,21,34,38,42,46,64,72,80,88,96,104,112,120,130,134,138,142,161,165,177,181], \
                 '-b':[9,13,25,29,50,54,58,62,68,76,84,92,100,108,116,124,146,150,154,158,169,173,185,189], \
                 '+c':[2,6,10,14,33,37,49,53,66,70,74,78,97,101,113,117,128,136,144,152,160,168,176,184], \
                 '-c':[18,22,26,30,41,45,57,61,82,86,90,94,105,109,121,125,132,140,148,156,164,172,180,188], \
                 'v' :list(range(3, 192,4))}

    if points in range(0,5): shell['+c'] = [] ; shell['-c'] = []
    pts = np.zeros(n*4)

    for i in range(n*4):
        if i in shell['-a']   : pts[i] = -a
        elif i in shell['+a'] : pts[i] = a
        elif i in shell['-b'] : pts[i] = -b
        elif i in shell['+b'] : pts[i] = b
        elif i in shell['-c'] : pts[i] = -c
        elif i in shell['+c'] : pts[i] = c
        elif i in shell['v']  : pts[i] = v

    return n, pts

def lebedevAngularGrid(order):
    #construct coordinates of Lebedev shell

    if order == 14:

        sum = 2
        a = [0.0] * sum ; b = [0.0] * sum
        v = [0.6666666666666667e-1, 0.7500000000000000e-1]
        n = [0, 2]

    elif order == 50:

        sum = 4
        b = [0.0] * sum
        v = [0.1269841269841270e-1, 0.2257495590828924e-1, 0.2109375000000000e-1, 0.2017333553791887e-1]
        a = [0.0, 0.0, 0.0, 0.3015113445777636e+0]
        n = [0, 1, 2, 3]

    elif order == 86:

        sum = 5
        b = [0.0] * sum
        v = [0.1154401154401154e-1, 0.1194390908585628e-1, 0.1111055571060340e-1, 0.1187650129453714e-1, 0.1181230374690448e-1]
        a = [0.0, 0.0, 0.3696028464541502e+0, 0.6943540066026664e+0, 0.3742430390903412e+0]
        n = [0, 2, 3, 3, 4]

    elif order == 302:

        sum = 12
        v = [0.8545911725128148e-3, 0.3599119285025571e-2, 0.3449788424305883e-2, 0.3604822601419882e-2, 0.3576729661743367e-2, \
             0.2352101413689164e-2, 0.3108953122413675e-2, 0.3650045807677255e-2, 0.2982344963171804e-2, 0.3600820932216460e-2, \
             0.3571540554273387e-2, 0.3392312205006170e-2]
        a = [0.0,0.0,0.3515640345570105,0.6566329410219612,0.4729054132581005,0.9618308522614784e-1,0.2219645236294178, \
             0.7011766416089545, 0.2644152887060663, 0.5718955891878961, 0.2510034751770465, 0.1233548532583327]
        b = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.8000727494073952,0.4127724083168531]
        n = [0,2,3,3,3,3,3,3,4,4,5,5]

    returned_order = [0]*sum ; g = np.empty((0, 4))
    for i in range(sum):
        returned_order[i], lg = ohSymmetry(n[i], a[i], b[i], v[i])
        g = np.vstack((g, lg.reshape(returned_order[i],4)))

    return np.sum(returned_order), g

def buildAtomicGrids(molecular_atoms, grid_mesh, methods):
    #get the atomic grids

    #get unique atom types - hydrogens last
    atomic_species = sorted(set(molecular_atoms), key = lambda i: period_table[i])[::-1]

    #get number of radial and angular points for grid type
    radial, angular, lebedev = getGridMetrics(grid_mesh)

    #define table of unique atom species in molecule - coordinates and weights
    atom_table = {}

    #loop over different atomic species
    for atom in atomic_species:

        if not atom in atom_table:
            period = period_table[atom]

            #radial point count
            r_points = radial[period]
            r, dr = methods['radial grid'](atom, r_points)

            #radial weights
            w = 4.0 * np.pi * r * r * dr

            #prune radial grid
            lebedev_points = lebedev[angular[period]]
            a_points = methods['radial prune'](lebedev_points, r_points)

            coordinates    = []
            volume_weights = []
            #angular grids
            for n in sorted(set(a_points)):

                #get shell points
                mesh = np.empty((n, 4))
                points, mesh_weights = lebedevAngularGrid(n)

                assert points, n

                #get indices of this n in r_points
                indices = np.where(a_points == n)[0]

                ##combine radial and angular grid
                coordinates.append(np.einsum('i,jk->jik', r[indices], mesh_weights[:,:3]).reshape(-1,3))
                volume_weights.append(np.einsum('i,j->ji', w[indices], mesh_weights[:,3]).ravel())

            #add atom species coordinates and weights to molecular species stack
            atom_table[atom] =  (np.vstack(coordinates), np.hstack(volume_weights))

    return atom_table

def general_partition(mol, coordinates, n_atoms, n_grids, radial_adjust, radii, becke):
    #main partition routine and radial adjust

    if radial_adjust != None :
        a = radial_adjust(mol, radii)
        adjust = lambda i, j, g : g + a[i,j]*(1.0 - g**2)


    meshes = np.empty((n_atoms, n_grids))
    separations = atom_separations(mol)

    for i in range(n_atoms):

        c = coordinates - mol[i].center
        meshes[i] = np.sqrt(np.einsum('ij,ij->i', c, c))

    becke_partition = np.ones((n_atoms, n_grids))
    for i in range(n_atoms):
        for j in range(i):

            g = (1/separations[i,j]) * (meshes[i] - meshes[j])

            #final radial adjustment
            if radial_adjust != None: 
                g = adjust(i, j, g) 

            g = becke(g)

            becke_partition[i] *= 0.5 * (1.0 - g)
            becke_partition[j] *= 0.5 * (1.0 + g)

    return becke_partition


def grid_partition(mol, atom_grid_table, radial_adjust, radii, becke):
    #partition molecular grid for DFT integration

    molecule_coordinates = np.array([i.center for i in mol])
    n_atoms = len(mol)

    coordinates = []
    weights     = []

    for ia in range(n_atoms):
        atom_coordinates, atom_volume = atom_grid_table[mol[ia].symbol] 

        #translate to center on atom
        atom_coordinates = atom_coordinates + mol[ia].center

        #do Becke partition
        n_grids = atom_coordinates.shape[0]
        becke_partition = general_partition(mol, atom_coordinates, n_atoms, n_grids, radial_adjust, radii, becke)

        #get weights
        w = atom_volume * becke_partition[ia] * 1.0/becke_partition.sum(axis=0)

        #add this atom to molecular coordinates and weights
        coordinates.append(atom_coordinates)
        weights.append(w)

    coordinates = np.vstack(coordinates)
    weights     = np.hstack(weights)

    return coordinates, weights

def GRID(mol, mesh):
    #grid driver

    atoms = [i.symbol for i in mol]

    methods = {'radial grid': muraKnowlesRadialGrid, 'radial prune': treutlerRadialPruning, \
               'radial adjust' : treutlerAdjust, 'radii': 'BRAGG', 'becke scheme': stratmann}

    #get shells for unique atomic species
    atom_grid_table = buildAtomicGrids(atoms, mesh, methods)

    #build shells for molecule
    coordinates, weights = grid_partition(mol, atom_grid_table, methods['radial adjust'], \
                                          methods ['radii'], methods['becke scheme'])

    return coordinates, weights
