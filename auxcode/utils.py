## Descrizione del file:
##
## Funzioni di utilità generale, non vincolate ad un ambito specifico.

import numpy as np
from numpy.linalg import cholesky
from scipy.special import roots_hermite



def axis_to_triu(arr, n = None, axis = -1, k = 0):
    '''
    Trasforma un asse dell'array in una matrice triangolare superiore creando un nuovo asse.
    I numeri del vettore vengono collocati nella una matrice triangolare superiore a partire
    dalla diagonale k-esima, come segue: ((1,k+1), (1,k+2), ..., (2,k+2), (2,k+3), ..., (n-k,n)).
    
    Parametri
    ---------
    arr : np.array, shape=(..., (n-k)*(n-k+1)//2, ...)
        Array da trasformare.
        La lunghezza dell'asse da espandere deve essere nella forma scritta sopra.
    n : int, default=None
        Dimensione della matrice triangolare.
        Se non è fornita, la dimensione della matrice viene calcolata a partire da quella dell'asse
        da espandere. Produce un errore se le dimensioni non lo permettono.
        Si tenga presente che il calcolo è un processo lento.
    axis : int, default=-1
        Asse dal quale prendere i valori da disporre sulla matrice triangolare.
    k : int, default=0
        Diagonale sulla quale iniziare a mettere i numeri.
        Non viene fatto alcun controllo sul valore k, potrebbe produrre un errore se viene preso
        fuori dall'intervallo della matrice.
    
    Restituisce
    -----------
    np.array, shape=(..., n, n, ...)
        Array con una dimensione in più, adiacente a quella espansa.
        Nota: crea un nuovo array, non modifica quello vecchio.
    '''
    
    # calcola la dimensione della matrice risultante
    if n is None:
        try:
            dimflt = np.polynomial.polynomial.Polynomial((-2*arr.shape[axis], 1, 1)).roots().max() + k
            n = np.int_(dimflt)
            assert n == dimflt
        except AssertionError:
            raise ValueError(f"La dimensione dell'asse {axis} non è corretta per k={k}!")
    
    # costruisce il nuovo array
    try:
        assert -arr.ndim <= axis < arr.ndim
        narr = np.zeros( arr.shape[:axis] + (n,n) + (arr.shape[axis+1:] if axis != -1 else ()) )
    except AssertionError:
        raise ValueError(f"L'indice dell'asse {axis} è fuori dall'intervallo possibile: arr.ndim={arr.ndim}!")
   
    # riempie l'array
    narr[( *[slice(None)]*len(arr.shape[:axis]), *np.triu_indices(n, k), ... )] = arr
    
    return narr


def triu_to_axis(arr, axis = -2, k = 0):
    '''
    Rimpiazza nell'array due assi adiacenti della stessa lunghezza con il vettore degli
    elementi triangolari superiori della matrice data da questi assi.
    I numeri del vettore vengono prelevati dalla matrice triangolare superiore a partire
    dalla diagonale k-esima, come segue: ((1,k+1), (1,k+2), ..., (2,k+2), (2,k+3), ..., (n-k,n)).
    
    Parametri
    ---------
    arr : np.array, shape=(..., n, n, ...)
        Array da trasformare.
        I due assi da sostituire devono avere la stessa lunghezza, cioè la relativa matrice
        deve essere quadrata.
    axis : int, default=-2
        La matrice considerata è data da questo asse e dal successivo.
    k : int, default=0
        Diagonale dalla quale iniziare a prendere i numeri.
        Non viene fatto alcun controllo sul valore k, potrebbe produrre un errore se viene preso
        fuori dall'intervallo della matrice.
    
    Restituisce
    -----------
    np.array, shape=(..., (n-k)*(n-k+1)//2, ...)
        Array con i due assi rimpiazzati da uno solo.
    '''
    
    try:
        assert arr.ndim >= 2
    except AssertionError:
        raise ValueError(f"L'array deve avere almeno dimensione 2!")
        
    try:
        assert -arr.ndim <= axis < arr.ndim - 1
    except AssertionError:
        raise ValueError(f"L'indice dell'asse {axis} è fuori dall'intervallo possibile: arr.ndim={arr.ndim}!")
    
    try:
        assert arr.shape[axis] == arr.shape[axis+1]
    except AssertionError:
        raise ValueError(f"I due assi {axis} e {axis+1} hanno lunghezze diverse!")
        
    return arr[( *[slice(None)]*len(arr.shape[:axis]), *np.triu_indices(arr.shape[axis], k), ... )]


def perturba_array(rng, eps, *arrs):
    '''
    Perturba uniformemente degli array.
    
    Parametri
    ---------
    rng : numpy.random.Generator
    	Generatore di numeri random
    eps : float
        Entità della perturbazione: ogni elemento subirà un cambiamento in [-eps, eps].
    *arrs : np.array
        Array da perturbare.
        
    Restituisce
    -----------
    np.array oppure tuple(np.array)
        Tupla di array perturbati. Stessa dimensione di quelli originari.
        Se è uno solo non lo incapsula nella tupla.
    '''
    
    parrs = [ arr + rng.uniform(-eps, eps, arr.shape) for arr in arrs ]
    
    if len(parrs) == 1:
        return parrs[0]
    else:
        return tuple(parrs)


def expect_multinormal_gausshermite(fun, n, mu = None, cov = None, maxpexp = 5, deg = None):
    '''
    Calcola il valore atteso di una funzione rispetto ad una distribuzione gaussiana multidimensionale.
    Usa la quadratura di Gauss-Hermite.
    
    Parametri
    ---------
    fun : callable
        Funzione della quale calcolare il valore atteso.
        Dev'essere vettoriale: prende in input una matrice che ha i punti elencati lungo l'ultimo asse.
        Restituisce un array con la prima dimensione invariata e nelle successive il risultato.
        Si tenga presente che viene chiamata una volta sola.
    n : int
        Numero di argomenti della funzione fun e, di conseguenza, dimensione della gaussiana.
    mu : np.array, size=n, default=None
        Vettore delle medie della gaussiana.
        Di default è il vettore nullo.
    cov : np.array, shape=(n,n), default=None
        Matrice di covarianza della gaussiana, deve avere la diagonale non-negativa.
        Di default è la matrice identità.
    maxpexp : int, default=5
        Controlla il numero massimo di punti da utilizzare per la quadratura.
        Il grado dei polinomi di Hermite da utilizzare, cioè la finezza dell'interpolazione,
        viene scelto il più alto possibile col vincolo deg^n <= 10^maxpexp, e comunque >= 2.
        Se il grado viene specificato direttamente questo parametro non ha effetto.
    deg : float, default=None
        Grado dei polinomi di Hermite da utilizzare.
        Se questo parametro è None viene ignorato e si considera maxpexp al suo posto.
    
    Restituisce
    -----------
    np.array
        Integrale vettoriale di fun.
        
    Nota
    ----
    Il metodo effettua una quadratura con deg^n punti, quindi basta poco per impallare tutto!
    '''
    
    if mu is None:
        mu = np.zeros(n)
    if cov is None:
        cov = np.eye(n)
    if deg is None:
        # calcolo deg affinché deg^n <= 10^maxpexp
        deg = int( max(2, np.power(10, maxpexp/n)) )
        #print(f"Quadratura di grado {deg}.") # DEBUG
    if deg < 2:
        raise ValueError(f"Il grado della quadratura non può essere minore di 2!")
        
    # calcola i punti e i coefficienti per la quadratura
    x, w = roots_hermite(deg)
    
    if n > 1:
        # crea l'array con le sequenze di punti
        xmesh = np.meshgrid(*([x] * n), indexing = 'ij')
        xs = np.dstack([ cd.flatten() for cd in xmesh ]).squeeze() # ATTENZIONE: non funziona per deg==1!

        # crea l'array con le sequenze di coefficenti
        wmesh = np.meshgrid(*([w] * n), indexing = 'ij')
        ws = np.dstack([ cd.flatten() for cd in wmesh ]).squeeze() # ATTENZIONE: non funziona per deg==1!
        ws = np.prod(ws, -1) # non servono separati, basta il prodotto
    else:
        xs, ws = x, w # l'integrazione è unidimensionale
    
    # calcolo la matrice di cambio di coordinate una volta sola
    if np.count_nonzero(cov - np.diag(np.diagonal(cov))) == 0: # matrice diagonale, gaussiane indipendenti
        L = np.sqrt(2 * np.diagonal(cov))
        fs = fun(xs * L.T + mu) if n > 1 else fun((xs * L.T + mu)[:,np.newaxis])
    else: # qui n > 1 per forza
        L = np.sqrt(2) * np.linalg.cholesky(cov)
        fs = fun(np.dot(xs, L.T) + mu)
    
    return np.sum(np.moveaxis(fs, 0, -1) * ws, axis = -1) * np.power(np.pi, -n/2)


def expect_multinormal_montecarlo(fun, n, rng, mu = None, cov = None, resol = 1e3):
    
    if mu is None:
        mu = np.zeros(n)
    if cov is None:
        cov = np.eye(n)
    resol = np.int_(resol)
    
    xs = np.stack([ rng.normal(mu[i], np.sqrt(cov[i,i]), size = resol) for i in range(n) ], 1)
    
    return np.mean(fun(xs), axis = 0)