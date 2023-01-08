import numpy as np

from .utils import axis_to_triu



def config_mesh(n):
    '''
    Restituisce la matrice di tutte le configurazioni con un certo numero di spin.
    
    Parametri
    ---------
    n : int
        Numero di spin.
        
    Restituisce
    -----------
    np.array({-1,+1}), shape=(2^n, n)
        Configurazioni dicotomiche con n spin.
    '''
    
    return np.array(np.meshgrid( *[[1,-1]]*n, indexing = 'ij' )).reshape(n,-1).T


def hamiltoniana(coeff, config):
    '''
    Calcola le hamiltoniane coi coefficienti dati su ogni configurazione.
    
    Parametri
    ---------
    coeff : np.array(float), shape=(r, n*(n+1)//2)
        Coefficienti dell'hamiltoniana, senza il termine -1 che va davanti a tutto.
        La funzione inferisce il numero di spin n dall'ultima dimensione dell'array config.
        I primi coefficienti sono relativi agli spin singoli, poi alle coppie, nell'ordine:
        1, 2, ..., n, (1,2), (1,3), ..., (2,3), (2,4), ..., (n-1,n)
        Nota: tutte le hamiltoniane devono avere lo stesso numero di spin.
    config : np.array({+1,-1}), shape=(m, n)
        Lista di configurazioni di spin.
        Ciascuna configurazione si sviluppa sull'ultimo asse.
    
    Restituisce
    -----------
    np.array(float), shape=(r, m)
        Valore dell'energia calcolata per ciascuna configurazione di spin.
    '''
    
    coeff, config = np.atleast_2d(coeff), np.atleast_2d(config)
    
    n_spin = config.shape[-1] # numero di spin
    
    # parte di campo esterno
    cest = np.tensordot(coeff[...,:n_spin], config, ((-1), (-1))) # così non devo trasporre niente
    
    # costruisco le interazioni come diag(C * I * C^T), dove ^T indica la trasposta
    # axis_to_triu mi serve a creare la matrice delle interazioni
    inter = np.einsum('ij,...jk,ik->...i', config, axis_to_triu(coeff[...,n_spin:], n = n_spin, k = 1), config,
                     optimize = True)
    
    return -( cest + inter )


def media_bg(beta, n, ham, fun):
    '''
    Calcola la media termica di alcune funzioni rispetto alle hamiltoniane date.
    
    Parametri
    ---------
    beta : float (positivo)
        Temperatura inversa
    n : int (positivo)
        Numero di spin del sistema.
        Le hamiltoniane e le funzioni devono prendere in input un array con n elementi.
    ham : np.array(float), shape=(r, n*(n+1)//2)
        Coefficienti delle hamiltoniane, come descritto nella funzione 'hamiltoniana'.
    fun : callable
        Funzione di spin a valori vettoriali della quale fare la media.
        
    Restituisce
    -----------
    media : np.array(float), shape=(r, ...)
        Medie termiche della funzione data rispetto alle varie hamiltoniane.
        Il primo asse indicizza le hamiltoniane, gli altri la media (la cui
        dimensione sarà quella del codominio della funzione).
    '''
    
    configs = config_mesh(n)
    
    fs = np.apply_along_axis(fun, 1, configs)
    
    # calcola le probabilità
    eng = hamiltoniana(ham, configs)
    w = np.exp(beta * (np.amin(eng, axis = -1, keepdims = True) - eng))
    cfg_ax = w.ndim - 1 # dimensione di w lungo la quale stanno le diverse configurazioni
    w = np.expand_dims( w, axis = tuple(range(-1, -fs.ndim, -1)) ) # versione espansa x broadcast

    return np.sum(fs * w, axis = cfg_ax) / w.sum(axis = cfg_ax)


