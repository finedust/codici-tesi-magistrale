import numpy as np
from scipy.optimize import root_scalar

from .meccstat import config_mesh
from .utils import triu_to_axis



def nparam(p):
	'''
	Calcola il numero di parametri q che avrà il sistema, in base ai nodi nascosti.
	
	Parametri
	---------
	p : int
		Numero di nodi nascosti.
		
	Restituisce
	-----------
	int
		Numero dei parametri (che va raddoppiato se si vogliono considerare anche i coniugati).
	'''
	
	p = np.int_(p)
	
	return (5*p - 1)*p//2


def separa_parametri(q, p):
	'''
	Separa il vettore dei parametri q in qs, qd, qu.
	
	Parametri
	---------
	q : np.array(float)
		Vettore dei parametri, ordinato come descritto sopra.
	p : int
		Numero di nodi nascosti.
		Il vettore q deve essere lungo nparam(p). Non dà errore in caso di lunghezza sbagliata!
	
	Restituisce
	-----------
	qs : np.array(float)
		Parametri stella, le magnetizzazioni di Mattis
	qd : np.array(float)
		Parametri tra repliche diverse
	qu : np.array(float)
		Parametri tra una stessa replica
	'''
	
	p = np.int_(p)
	psq = p**2
	
	return q[:psq], q[psq:2*psq], q[2*psq:]


def genera_cfg_ortogonali(N, P, ort = None, met = 'rnd', rng = None, eps = 0):
    
    if N%2 != 0:
        raise ValueError(f"Non ci sono vettori ortogonali in dimensione {N} perché è dispari!")
    
    if ort is None:
        if rng is None:
            rng = np.random.default_rng()
        ort = np.atleast_2d( 2 * rng.integers(0, 2, N) - 1 )
    else:
        ort = np.atleast_2d(ort)
        
    if P < ort.shape[0]:
        print(f"Sono stati richiesti meno vettori di quelli già disponibili!")
        return
    
    if met == 'ord':
        cfg = np.full(N, 1)
    elif met == 'rnd':
        if rng is None:
            rng = np.random.default_rng()
        cfg = 2 * rng.integers(0, 2, N) - 1
    else:
        raise ValueError(f"Metodo errato, possibilità: 'ord' oppure 'rnd'.")
    #print(f"Procedura {'ordinata' if met == 'ord' else 'casuale'},",
    #     f"configurazione di partenza {cfg}")
        
    tol = N * eps
    
    num = 1
    try:
        while (met != 'ord' or num < 2**N) and ort.shape[0] < P:
            if num > 0: # non cambia il primo perché è già stato impostato
                if met == 'ord':
                    idx = np.argmax(cfg) # trova il primo +1
                    cfg[idx] *= -1
                    cfg[:idx] = 1
                elif met == 'rnd':
                    idx = rng.integers(0, N)
                    cfg[idx] *= -1
            
            good = True
            for x in ort:
                if np.abs(np.sum(x * cfg)) > tol:
                    #print(f"Tentativo {cfg} non ortogonale a {x}")
                    good = False
                    break
            if good:
                #print(f"Aggiungo {cfg}")
                ort = np.vstack( (ort, cfg) )
                
            num += 1
    except KeyboardInterrupt:
        pass
    
    return ort

    #def int_to_cfg(N, num): return np.where(num & (1 << np.arange(N-1, -1, -1)) == 0, -1, 1)
    #def cfg_to_int(N, cfg): return np.sum(np.where(cfg == -1, 0, 1) * (1 << np.arange(N-1, -1, -1)))


def dist_xis(P, q):
    '''
    Costruisce una distribuzione su {+1,-1}^P con media degli spin nulla e correlazioni q.
    
    Parametri
    ---------
    P : int
        Numero di spin.
    q : float([0,1])
        Correlazione richiesta.
        Per altri valori di q (ad esempio negativi) non è detto che si riesca a trovare una
        distribuzione appropriata perché il sistema può presentare frustrazione.
    
    Restituisce
    -----------
    np.array(float)
        Vettore di probabilità che rappresenta la distribuzione, per le configurazioni ordinate
        come in config_mesh.
    '''
    
    mesh = config_mesh(P)
    
    if np.isclose(q, 1, atol = 1e-6, rtol = 1e-6): # costruisce una delta nelle due configurazioni estremali
        return np.where( np.abs(np.sum(mesh, axis = -1)) == P , 0.5, 0)
    
    if np.isclose(q, 0, atol = 1e-6, rtol = 1e-6): # fornisce una distribuzione uniforme
        return np.ones(2**P) / 2**P
    
    if q < 0 or q > 1:
        raise ValueError(f"A priori non è detto che si possa trovare una distribuzione per q={q}")
        
    sq = np.sum(mesh, axis = -1)**2
    corrf = mesh[:,0] * mesh[:,1]
    def equaz(J, P, q):
        cham = J/2 * sq
        ham = np.exp(cham - np.amax(cham))
        return np.sum(ham * corrf)/np.sum(ham) - q

    J = root_scalar(equaz, bracket = (-10,10), args = (P, q)).root
    cham = J * np.sum( triu_to_axis(mesh[:, :, np.newaxis] * mesh[:, np.newaxis, :], k = 1) , axis = -1)
    return np.exp(cham - np.amax(cham))


