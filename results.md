![image](https://user-images.githubusercontent.com/73105740/142720205-ddb5a6ad-4d6c-4a1a-8171-9ea1af5eece4.png)
## Density Function Theory Program - kspy-tddft(tda) results
   The output from the SCF part of the program is 

	      ks output
	molecule is             water
	geometry is             OH bonds - 0.96      HOH angle - 104.42
	basis is                STO-3G {psi4 format}
	analytic integration    aello cython - McMurchie-Davidson scheme
	numerical integration   (Mura-Knowles, Lebedev)
	                        radial prune is Aldrich-Treutler
	                        Becke Partition scheme is Stratmann 
	                        Radial adjust is Treutler
	                        order: period 1 (10,11) and period 2 (15,15)
	mesh                    close
	functional              RPA
	diis                    True   buffer size is  6

	scf control             maximum cycles are  50         convergence tolerance  1e-06

	 cycle     1 electron         coulomb         exchange          electrons
	                                                                                   ΔE         diis norm
	-------------------------------------------------------------------------------------------------------
	    0     -127.35805722     54.98251023     -9.81293954         10.0000 

	    1     -117.59033633     42.89506750     -8.53626865         10.0000 
	                                                                                 8.002968      0.917212 
	    2     -125.68328911     51.55312723     -9.36607660         10.0000 
	                                                                                 5.182177      0.755499 
	    3     -122.66389204     47.65047850     -8.93676745         10.0000 
	                                                                                 1.302286      0.060478 
	    4     -122.43525249     47.39500533     -8.91262519         10.0000 
	                                                                                 0.090827      0.007214 
	    5     -122.40802100     47.36478049     -8.90966476         10.0000 
	                                                                                 0.010677      0.000915 
	    6     -122.40473419     47.36113972     -8.90931134         10.0000 
	                                                                                 0.001170      0.000151 
	    7     -122.40406831     47.36040214     -8.90923965         10.0000 
	                                                                                 0.000303      0.000003 
	    8     -122.40408162     47.36041688     -8.90924108         10.0000 
	                                                                                 0.000006      0.000000 
	    9     -122.40408162     47.36041688     -8.90924108         10.0000 

	final energies (Hartree)
	------------------------
	one electron        -122.4040816204
	coulomb               47.3604168778  
	exchange              -8.9092410801  
	nuclear repulsion      9.1882584177   
	total electronic     -83.9529058227 

	final total energy   -74.7646474050 

	molecular orbitals
	------------------
	 0  -18.27186 occupied  
	 1   -0.83468 occupied  
	 2   -0.38648 occupied  
	 3   -0.15301 occupied  
	 4   -0.06096 homo      
	 5    0.31525 lumo      
	 6    0.42301 virtual   

	mulliken populations
	--------------------
	 0    1.99686   1s  
	 1    1.84752   2s  
	 2    2.00000   2px 
	 3    1.08767   2py 
	 4    1.45385   2pz 
	 5    0.80705   1s  
	 6    0.80705   1s  

	atomic charge
	---------------
	 0   -0.38590 O   
	 1    0.19295 H   
	 2    0.19295 H   

	dipole momemts (Debye)
	----------------------
	 x= 0.00000  y= 0.00000  z= 1.73697 

The output from the **TDA_properties** class with roots=1  and excitation='singlet' is

	TDA analysis for singlet states
	--------------------------------------------------------------------------
	root 1     energy 11.5452 eV   107.39 nm  f = 0.0026  
	principal excitation 4->5      magnitude  -0.7071 
	transition electric dipole (length gauge)(au)    -0.0965  -0.0000   0.0000
	                           norm² 0.0093   oscillator strength 0.0026  

	transition electric dipole (velocity gauge)(au)   0.1288  -0.0000   0.0000
	                           norm² 0.0166   oscillator strength 0.0261  

	transition magnetic dipole (length gauge)(au)    -0.0000  -0.4876  -0.0000

	rotary strengths           (length) 0.0000      (velocity) -0.0000 

The output from  the **TDA_properties** class for the transition_NO method with root=0 and threshold=0.2. The threshold is the level below which principal transition will not be reported.

	Transition Natural Orbitals
	---------------------------
	Natural transition orbitals for state 1   11.5452 eV  maximum component is 1.0000  
	Inferred transition 
	4 -> 5   (1.0000 -> 1.0000)

The **transition_properties** method of the **TDA_properties** class returns a dictionary with keys as follows\
'i' - root, 't' - excitation type (singlet or triplet), 'w' - wavenumber, e' - excitation energy, 'j' - principal jump, 'd' - transition electric dipole length gauge, 'od' - transition electric dipole length gauge oscillator strength, 'sd' - square of transition electric dipole length gauge, 'n' - transition electric dipole velocity gauge, 'sn' - square of transition electric dipole velocity gauge, 'on' - transition electric dipole velocity gauge oscillator strength, 'a' - transition magnetic dipole length gauge, 'rl' - rotary strength length gauge, 'rv' - rotary strength velocity gauge.

The output from the **TDA_properties** module for the **spectrum** method with roots=8, excitation='singlet' and type='length' is

![image](https://user-images.githubusercontent.com/73105740/149924578-81ff2639-8498-4503-ab4b-8a8ad491336f.png)

To run the **RT_TDA** class you must supply ks data, field data and run parameters to the class as dictionaries. The ks data is shown in the code. The field is described by keys for the external field application type ('kick' or 'gaussian'), it's intensity and the center of the applied pulse. For a Gaussian pulse additionally the 'spread' rho must be specified. The run parameters are the number of steps, the time of one step, the units for time and dipole as a list and the gauge origin (default coordinate origin). So as an example {'steps':1000, 'tick':0.2, 'units':['au','debye'], 'gauge':'charge center'} will run 1000 steps of 0.2 atomic time units each ie 200 atu in the charge center gauge. The **magnus** method then runs a 2nd order Magnus expansion time propogation, the method takes a single parameter which is list of the dipole directions to be calculated. The resultant propogation can then be visualised with the **display** method which shows the propogation in each of the specified axis directions. This is for {'steps':1000, 'tick':0.2, 'units':['au','debye'], 'gauge':'origin'}

![image](https://user-images.githubusercontent.com/73105740/151696742-c06900f4-6e94-439d-9178-1a7059a25b7a.png)

Running **spectrum** for {'damping':5000, 'eV range':1.5, 'resolution':0.0001, 'function':np.abs} on the above dipole propogation yields

![image](https://user-images.githubusercontent.com/73105740/151693932-af13d66f-e356-42ae-91d8-3494c45e2346.png)

The agreement with the analytical LR-TDA results is reasonable, the actual peak values are returned as a dictionary by the **spectrum** method (keys 'x', 'y' and 'z') and also printed to the console

    RT-TDA dipole transition energies (eV)
    --------------------------------------
    axis- x       11.47  
    axis- y       17.64  21.69  36.22  
    axis- z       14.03  26.60  36.60 

A **get_observables()** method is provided which returns a dictionary of the dipole propogations with keys 'time', 'field', 'x', 'y' and 'z' (if the corresponding dipole component has been selected in **magnus**).

