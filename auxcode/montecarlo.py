import numpy as np

from .utils import axis_to_triu



def deng_hamiltoniana(cfgs, idx, ham):
    '''
    Calcola la differenza energetica tra due stati che differiscono per uno spin.
    Fa questo data una certa hamiltoniana di cui siano noti i coefficienti e per ogni configurazione data.
    Il calcolo è H(flip_i(sigma)) - H(sigma) dove flip_i è l'inversione dello spin i-esimo.
    
    Parametri
    ---------
    cfgs : np.array({+1,-1}), shape=(num, N)
        Le configurazioni iniziali, ciascuna con N spin; sopra erano chiamate sigma.
    idx : np.array({0, ..., N-1}), shape=(num,)
        L'indice da invertire in ciascuna configurazione; sopra detto i.
    ham : np.array(float)
        Coefficienti dell'hamiltoniana, ordinati come già descritto e senza il -1 iniziale.
        
    Restituisce
    -----------
    np.array(float), shape=(num,)
        Differenza energetica per ciascuna configurazione.
    '''
    
    shape = cfgs.shape
    inter = axis_to_triu(ham[shape[1]:], shape[1], k = 1)
    
    dinter = np.zeros(shape[0])
    for k in range(shape[0]):
        i = idx[k]
        dinter[k] = np.dot(inter[:i,i], cfgs[k,:i]) + np.dot(inter[i,i+1:], cfgs[k,i+1:])
    
    return 2 * cfgs[np.arange(shape[0]), idx] * (ham[idx] + dinter)
    # Ricorda: i coefficienti sono senza il -1 quindi qui alla fine cambio segno


def metropolis(rng, N, beta, num, steps, deng, *dengargs):
    '''
    Esegue un campionamento dalla distribuzione di Boltzmann-Gibbs usando l'algoritmo Metropolis-Hastings.
    La differenza energetica tra gli stati è calcolata tramite l'apposita funzione fornita.
    
    Parametri
    ---------
    rng : numpy.random.Generator
        Generatore di numeri casuali
    N : int
        Numero di spin dell'hamiltoniana
    beta : float
        Temperatura inversa
    num : int
        Numero di estrazioni da effettuare
    steps : int
        Iterazioni dell'algoritmo da effettuare su dei dati iniziali casuali uniformi.
    deng : callable
        Funzione che calcola il divario energetico tra la configurazione flip_i(sigma) e sigma.
        Qui flip_i indica il cambiamento dello spin i-esimo.
        La funzione deve accettare i parametri:
            np.array({+1,-1}), shape=(num,N)
                Configurazioni di partenza con N spin, sopra chiamate sigma
            np.array({0, ..., N-1}), size=num
                Indici da invertire, sopra chiamati i
        Eventuali altri parametri che la funzione richiede verranno passati tramite *dengargs.
        Deve restituire un array di lunghezza num.
    *dengargs
        Questi ulteriori parametri verranno passati alla funzione deng senza modifiche.
    
    Restituisce
    -----------
    np.array({+1,-1})
        Array delle estrazioni.
        Dimensione (num, N).
    '''
        
    # Questo serve se vogliamo fare gli scambi casualmente
    #idx = rng.integers(0, N, size = (num, steps)) # sono traslati di 1 a sx
    
    s = 2 * rng.integers(0, 2, size = (num, N)) - 1 # initial state
    
    for t in range(steps):
        #ic = idx[:,t]
        ic = ((t%N) * np.ones(num)).astype(int)
        
        diffeng = deng(s, ic, *dengargs)
        change = np.logical_or( diffeng <= 0, rng.random(size = (num)) < np.exp(-beta * (diffeng>0) * diffeng) )
        # Questa non credo ci vada, sarebbe per fare in modo che ciascuno stato abbia una
        #  probabilità non-nulla di rimanere invariato, assicurando così (forse) l'ergodicità
        #  della catena anche per beta>1
        #change = np.logical_or( change, rng.random(size = num) > 0.05 )
        changerows = np.arange(num)[change]
        s[changerows, ic[changerows]] *= -1
    return s