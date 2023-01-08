import numpy as np
from scipy.linalg import eigh

from time import perf_counter_ns

from .utils import expect_multinormal_gausshermite, triu_to_axis, axis_to_triu
from .auxiliaryfun import separa_parametri
from .meccstat import config_mesh, hamiltoniana, media_bg
from .montecarlo import metropolis



def L_coeff(q, lamb, xi, z):
    '''
    Calcola i coefficienti dell'hamiltoniana definita nella tesi.
    Inferisce il numero di spin (p) da xi.
    La funzione è vettoriale in xi e z.
    
    Parametri
    ---------
    q : np.array(float in [-1,1]), size=nparam(p)
        Parametri d'ordine ordinati come visto sopra.
    lamb : float
        Coefficiente, sarà uguale a 1 o beta^2.
    xi : np.array({+1,-1}), shape=(r, p)
        Parametri. Da questi inferisco il numero di spin dell'hamiltoniana.
    z : np.array(float), shape=(s, p(p+1)//2)
        Array di variabili gaussiane per l'integrazione
    
    Restituisce
    -----------
    np.array(float), shape=(r, s, p*(p+1)/2)
        Coefficienti dell'hamiltoniana L, come definiti nella funzione 'hamiltoniana' (senza il -1 davanti).
    '''
    
    # inizializzazione
    xi = np.atleast_2d(xi)
    z = np.atleast_2d(z)
    r, p, s = *xi.shape, z.shape[0]
    coeff = np.zeros((r, s, p*(p+1)//2)) # divisione intera, tanto non ha resto
    #print(p, q.size) # DEBUG
    
    # separo i parametri
    qs, qd, qu = separa_parametri(q, p)
    qd = (qd.reshape(p,p) + qd.reshape(p,p).T) / 2 # tanto non mi serve quella originaria
    qdv = triu_to_axis(qd, k = 1) # prende solo la parte triangolare superiore
    
    # calcolo coefficienti di interazione
    coeff[...,p:] = lamb * (qu - qdv)
            
    # calcolo coefficienti di campo esterno
    # ATTENZIONE: qui nei conti spuntano fuori cose strane, il valore assoluto nella radice non ci andrebbe!
    coeff[...,:p] = np.broadcast_to( (lamb * np.dot( xi, qs.reshape(p,p) ))[:,np.newaxis,:], (r, s, p) )
    
    lamb = np.sqrt(lamb) # non serve più quello vero
    
    coeff[..., :p] += lamb * z[:, :p] * np.sqrt(np.abs(2 * qd.diagonal() - np.sum(qd, axis = 0)))
    # l'asse della somma è indifferente per simmetria
    
    tmp = axis_to_triu( np.sqrt(np.abs(qdv)) * z[:, p:] , p, k = 1)
    coeff[..., :p] += lamb * np.sum(tmp + tmp.transpose(0,2,1), axis = 2)
    
    return np.squeeze(coeff)


def sol_nl(p, M, Mhat = None):
    '''
    Produce la soluzione semplificata delle condizioni stazionarie nel caso di pattern ortogonali.
    La soluzione semplificata è quella in cui le matrici dei parametri sono scalari con la stessa diagonale
    perché nel caso ortogonale siamo sulla linea di Nishimori.
    Analogamente le matrici coniugate avranno un loro valore.
    
    Parametri
    ---------
    p : int (positivo)
        Numero di nodi nascosti.
    M : float (in [-1,1])
        Sovrapponibilità richiesta per i parametri ordinari.
    Mhat : float (in [-1,1]), default=M
        Coefficiente per i parametri coniugati.
    
    Restituisce
    -----------
    q : np.array
        Vettore dei parametri, ordinato come specificato sopra.
    qhat : np.array
        Vettore dei parametri coniugati, stesso ordinamento.
    '''
    
    p = np.int_(p) # serve un casting perché ogni tanto viene passato come float
    if Mhat == None:
        Mhat = M
        
    q = np.concatenate(( M * np.eye(p).reshape(p**2), M * np.eye(p).reshape(p**2), np.zeros(p*(p-1)//2) ))
    qhat = np.concatenate(( Mhat * np.eye(p).reshape(p**2), Mhat * np.eye(p).reshape(p**2), np.zeros(p*(p-1)//2) ))
    
    return q, qhat


def calcola_cs_minimali(param, beta, alpha, deg = 30):
    '''
    Calcola il membro di destra delle condizioni stazionarie minimali.
    
    Parametri
    ---------
    param : np.array(float), size=2
        Contenente i parametri m,q o i corrispettivi coniugati.
    beta : float
        Temperatura inversa.
    alpha : float
        Rapporto tra dati e neuroni visibili.
    deg : int, default=30
        Grado dei polinomi di Hermite da calcolare.
        Viene passato al metodo expect_multinormal_gausshermite.
        
    Restituisce
    -----------
    np.array(float), size=2
        Contiene i due risultati calcolati che corrispondono a m,q (o i coniugati).
    '''
    
    def fun(z):
        # uso squeeze perché so che verrà integrata in dimensione 1 ma il parametro è 2-dim.
        val = np.tanh((beta**2) * param[0] + beta * np.sqrt(np.abs(param[1])) * z.squeeze())
        return np.stack( (val, np.power(val,2)), axis = 1)
    
    return  alpha * (beta**2) * expect_multinormal_gausshermite(fun, 1, deg = deg)


def calcola_cs_q(p, idx, qhat, xis_dist):
    '''
    Calcola il membro di destra delle condizioni stazionarie per i parametri q.
    
    Parametri
    ---------
    p : int
        Numero di nodi nascosti.
    idx : np.array(bool)
        Vettore che indica quali equazioni calcolare, relativamente al vettore dei parametri q.
    qhat : np.array(float)
        Vettore dei parametri completo, ordinati come sopra.
    xis_dist : np.array(float), size=2^p
        Distribuzione dei pattern originari, utilizzata per calcolare il valore atteso.
        Le relative configurazioni sono elencate nel seguente ordine (ottenuto da config_mesh):
        (1, ..., 1, 1), (1, ..., 1, -1), (1, ..., -1, 1), ..., (-1, ..., -1, -1)
        
    Restituisce
    -----------
    np.array(float)
        Vettore dei risultati calcolati, ordinati come in idx.
    '''
    
    # Verifico su quali indici devo calcolare le medie termiche
    idxs, idxd, idxu = separa_parametri(idx, p)
    idxs = idxs.reshape(p,p)
    idxd = idxd.reshape(p,p)
    #print(idxs,idxd,idxu) # DEBUG
    # indici per le medie singole
    sing = idxs.any(axis = 0) # necessari per qs
    sing = np.logical_or(sing, idxd.any(axis = 0)) # aggiungo quelli necessari per qd
    sing = np.logical_or(sing, idxd.any(axis = 1))
    #print(sing) # DEBUG
    
    # Funzione di spin per calcolare le medie termiche
    funspin = lambda x: np.concatenate( (x[sing], triu_to_axis(x*x[:,np.newaxis], k = 1)[idxu.astype(bool)]) )
    
    def argauss(zs):
        q = np.zeros((zs.shape[0], idx.sum())) # array dei risultati
        
        xis = config_mesh(p)
        medieter = media_bg(1, p, L_coeff(qhat, 1, xis, zs), funspin).transpose(1,0,2)
        
        i = 0 # coefficiente per scorrere q
        # calcolo i qs
        for (mu,nu) in zip(*idxs.nonzero()):
            # sum(sing[:,nu]) serve a vedere a quale indice di mediater sta nu perché può non averli tutti
            q[:, i] = np.average( xis[:, mu] * medieter[..., np.sum(sing[:nu])] , axis = 1, weights = xis_dist )
            i += 1
        # calcolo i qd
        for (mu, nu) in zip(*idxd.nonzero()):
            q[:, i] = np.average( medieter[..., np.sum(sing[:mu])] * medieter[..., np.sum(sing[:nu])] ,
                              axis = 1, weights = xis_dist )
            i += 1
        # inserisco i qu
        q[:, i:] = np.average( medieter[..., np.sum(sing):] , axis = 1, weights = xis_dist )
    
        return q
    
    return expect_multinormal_gausshermite(argauss, p*(p+1)//2)


def calcola_cs_qhat(p, idx, q, xis_dist, beta, alpha):
    '''
    Calcola il membro di destra delle condizioni stazionarie per i parametri coniugati qhat.
    
    Parametri
    ---------
    p : int
        Numero di nodi nascosti.
    idx : np.array(bool)
        Vettore che indica quali equazioni calcolare, relativamente al vettore dei parametri qhat.
    q : np.array(float)
        Vettore dei parametri coniugati completo, ordinati come sopra.
    xis_dist : np.array(float), size=2^p
        Distribuzione dei pattern originari, utilizzata per calcolare il valore atteso.
        Le relative configurazioni sono elencate nel seguente ordine:
        (1, ..., 1, 1), (1, ..., 1, -1), (1, ..., -1, 1), ..., (-1, ..., -1, -1)
    beta : float (positivo)
        Temperatura inversa.
    alpha : float (positivo)
        Rapporto tra dati e neuroni.
        
    Restituisce
    -----------
    np.array(float)
        Vettore dei risultati calcolati, ordinati come in idx.
    '''
    
    # Verifico su quali indici devo calcolare le medie termiche
    idxs, idxd, idxu = separa_parametri(idx, p)
    idxs = idxs.reshape(p,p)
    idxd = idxd.reshape(p,p)
    #print(idxs,idxd,idxu) # DEBUG
    # indici per le medie singole
    sing = idxs.any(axis = 0) # necessari per qs
    sing = np.logical_or(sing, idxd.any(axis = 0)) # aggiungo quelli necessari per qd
    sing = np.logical_or(sing, idxd.any(axis = 1))
    #print(sing) # DEBUG
    
    # Funzioni di spin per calcolare le medie termiche
    fun_spinlq = lambda x: np.concatenate( (x[sing], triu_to_axis(x*x[:,np.newaxis], k=1)[idxu.astype(bool)]) )
    fun_spinmu = lambda y: triu_to_axis(y*y[:,np.newaxis], k=1)[idxu.astype(bool)]
    
    def argauss(ws):
        qhat = np.zeros((ws.shape[0], idx.sum()))
        
        # Costruisco i coefficienti di Mstar
        xis = config_mesh(p)
        ham_mstar = np.concatenate(( np.zeros(p),
                                    np.average(triu_to_axis( xis[:,:,np.newaxis]*xis[:,np.newaxis,:] , k = 1) ,
                                               axis = 0, weights = xis_dist)
                                   ))
        # costruisco anche le probabilità di BG associate per fare tutto in un passaggio
        # altrimenti poi dovrei costruire medieter a pezzi con una doppia chiamata a medie_bg
        eng_star = hamiltoniana(ham_mstar, xis).squeeze() # ho una sola hamiltoniana
        w_star = np.exp(beta * (np.amin(eng_star) - eng_star)) # non sono normalizzati ma non importa
        
        # uso sempre xis per evitare doppioni, ma non c'entra con quello sopra!
        medieter = media_bg(1, p, L_coeff(q, beta**2, xis, ws), fun_spinlq).transpose(1,0,2) # metto le ws davanti
        
        i = 0 # coefficiente per scorrere qhat
        # calcolo i qs
        for (mu,nu) in zip(*idxs.nonzero()):
            qhat[:, i] = np.average( xis[:, mu] * medieter[..., np.sum(sing[:nu])] , axis = 1, weights = w_star)
            i += 1
        # calcolo i qd
        for (mu, nu) in zip(*idxd.nonzero()):
            qhat[:, i] = np.average( medieter[..., np.sum(sing[:mu])] * medieter[..., np.sum(sing[:nu])] ,
                                    axis = 1, weights = w_star)
            i += 1
        # inserisco i qu
        qhat[:, i:] = np.average( medieter[..., np.sum(sing):] , axis = 1, weights = w_star)
    
        return qhat
    
    qhat = expect_multinormal_gausshermite(argauss, p*(p+1)//2)
    
    # Calcolo il termine in più su qu e moltiplico per il coefficiente
    ham_mu = np.concatenate(( np.zeros(p), q[2*(p**2):] ))
    qhat[np.sum(idx[:2*(p**2)]):] -= media_bg(beta**2, p, ham_mu, fun_spinmu).squeeze()
    
    return alpha * (beta**2) * qhat


def deng_prob_diretto(cfgs, idx, beta, xi):
    '''
    Calcola la differenza energetica tra due stati che differiscono per uno spin.
    Questa funzione è specializzata sull'energia del problema diretto per RBM.
    Il calcolo è H(flip_i(sigma)) - H(sigma) dove flip_i è l'inversione dello spin i-esimo.
    
    Parametri
    ---------
    cfgs : np.array({+1,-1}), shape=(num, N)
        Le configurazioni iniziali, ciascuna con N spin; sopra erano chiamate sigma.
    idx : np.array({0, ..., N-1}), shape=(num,)
        L'indice da invertire in ciascuna configurazione; sopra detto i.
    beta : float
        Temperatura inversa.
    xi : np.array({+1,-1}), shape=(P, N)
        Array di sinapsi rispetto alle quali è condizionata la distribuzione di sigma.
    
    Restituisce
    -----------
    np.array(float), shape=(num,)
        Differenza energetica per ciascuna configurazione.
    '''
    
    # questo serve per numba
    #cfgs = cfgs.astype(np.float32)
    #xi = xi.astype(np.float32)
    
    # Usiamo l'equivalenza:
    # log(cosh(a+b)/cosh(a)) = |a+b|-|a|+log1p(e^-2|a+b|)-log1p(e^-2|a|)
    lamb = beta / np.sqrt(cfgs.shape[1])
    a = lamb * np.dot(cfgs, xi.T) # equivalente a np.matmul ma è supportato da numba
    apb = np.abs( a - 2 * lamb * cfgs[np.arange(cfgs.shape[0]), idx, np.newaxis] *
                 np.broadcast_to(xi, (cfgs.shape[0], *xi.shape))[np.arange(cfgs.shape[0]), :, idx] )
    a = np.abs(a)
    return - np.sum( apb - a + np.log1p(np.exp(-2*apb)) - np.log1p(np.exp(-2*a)) , axis = 1)


def deng_prob_inverso(xis, idx, beta, sigmas, tau_mesh):
    '''
    Calcola la differenza energetica tra due stati che differiscono per uno spin.
    Questa funzione è specializzata sull'energia del problema diretto per RBM.
    Il calcolo è H(flip_i(sigma)) - H(sigma) dove flip_i è l'inversione dello spin i-esimo.
    
    Parametri
    ---------
    xis : np.array({+1,-1}), shape=(num, N*P)
        Le configurazioni iniziali come vettore unidimensionale di N*P spin; sopra erano chiamate sigma.
    idx : np.array({0, ..., N*P-1}), shape=(num,)
        L'indice da invertire in ciascuna configurazione; sopra detto i.
        Siccome le configurazioni sono in riga anche l'indice è unidimensionale.
    beta : float
        Temperatura inversa.
    sigmas : np.array({+1,-1}), shape=(M, N)
        Array di esempi rispetto ai quali è condizionata la distribuzione di xi.
    tau_mesh : np.array({+1,-1}), shape=(2**P, P)
        Tutte le configurazioni con P elementi.
        Viene calcolato con config_mesh(P) a parte e poi viene passato alla funzione
        per risparmiare tempo.
        
    Restituisce
    -----------
    np.array(float), shape=(num,)
        Differenza energetica per ciascuna configurazione.
    '''
    
    num = xis.shape[0]
    M, N = sigmas.shape
    P = xis.shape[1] // N
    #print(num, N, M, P) # DEBUG
    
    xis = xis.reshape(xis.shape[0], P, N)
    idxp, idxn = idx//N, idx%N # coordinate dell'indice da aggiornare
    
    lamb = beta / np.sqrt(N)
    
    tmp = lamb * np.sum( xis[:, np.newaxis, ...] * tau_mesh[..., np.newaxis] , axis = -2) # shape=(num,2**P,N)
    tmpmod = tmp[np.arange(num), :, idxn] - (lamb * 2 
        * np.expand_dims(xis[np.arange(num), idxp, idxn], 1) 
        * np.broadcast_to(tau_mesh, (num, 2**P, P))[np.arange(num), :, idxp]) # shape=(num,2**P)
    #print(f"tmp: max={np.amax(tmp):e},min={np.amin(tmp):e} -- tmpmod: max{np.amax(tmpmod):e},min={np.amin(tmpmod):e}")
    cshtmp, cshtmpmod = np.cosh(tmp), np.cosh(tmpmod)
    #print(f"csh: max={np.amax(cshtmp):e},min={np.amin(cshtmp):e} -- cshm: max={np.amax(cshtmpmod):e},min={np.amin(cshtmpmod):e}")
    prodmsk = np.zeros((num, 2**P, N)).astype(bool)
    prodmsk[np.arange(num), :, idxn] = True
    prdpart = np.prod(np.where(prodmsk, 1, cshtmp), axis = -1)
    #print(f"prdpart: max={np.amax(prdpart):e},min={np.amin(prdpart):e}")
    cnum, cden = cshtmpmod * prdpart, cshtmp[np.arange(num), :, idxn] * prdpart
    #print(f"cnum: max={np.amax(cnum):e},min={np.amin(cnum):e} -- cden: max={np.amax(cden):e},min={np.amin(cden):e}")
    pt1 = M * ( np.log(np.sum(cnum, axis = 1)) - np.log(np.sum(cden, axis = 1)) )
    
    c = lamb * np.dot(xis[np.arange(num), idxp, :], sigmas.T)
    ctild = np.abs( c - 2 * lamb 
                   * xis[np.arange(num), idxp, idxn, np.newaxis] 
                   * np.broadcast_to(sigmas, (num, M, N))[np.arange(num), :, idxn] )
    c = np.abs(c)
    pt2 = np.sum( c - ctild + np.log1p(np.exp(-2*c)) - np.log1p(np.exp(-2*ctild)) , axis = 1)
    
    return pt1 + pt2


def estrazione_prob_diretto(rng, N, xi, beta, num, steps = 100):
    '''
    Esegue un campionamento relativo al problema diretto per la RBM.
    Estrae gli esiti dalla distribuzione diretta P(sigma|xi) con metodo MCMC.
    Per il momento funziona solo con Metropolis-Hastings.
    
    Parametri
    ---------
    rng : numpy.random.Generator
        Generatore di numeri casuali
    N : int
        Numero di nodi visibili
    xi : np.array({+1,-1}), shape=(P, N)
        Stato delle sinapsi da usare per l'estrazione
        P è il numero di unità nascoste.
    beta : float
        Temperatura inversa
    num : int
        Numero di estrazioni da effettuare
    steps : int
        Numero di iterazioni dell'algoritmo MCMC di campionamento
        
    Restituisce
    -----------
    np.array({+1,-1}), shape=(num,N)
        Array delle estrazioni.
    '''
    
    return metropolis(rng, N, 1, num, steps, deng_prob_diretto, beta, xi)


def estrazione_prob_inverso(rng, N, alphas, P, xis, beta, betainv = None, num = 1, steps = 100, debug = False):
    '''
    Esegue un campionamento relativo al problema inverso per la RBM.
    Prima estrae i sigma dalla dalld distribuzione diretta P(sigma|xis),
    poi fa lo stesso con l'inversa P(xi|sigma).
    Per il momento funziona solo con Metropolis-Hastings.
    
    Parametri
    ---------
    rng : numpy.random.Generator
        Generatore di numeri casuali
    N : int
        Numero di nodi visibili
    alphas : np.array(float)
        Carico del sistema: rapporto tra il numero di sigma estratti e N.
        Il problema inverso verrà risolto per ogni alpha nell'array.
    P : int
        Numero di nodi nascosti.
    xis : np.array({+1,-1}), shape=(P, N)
        Stato delle sinapsi originarie, da usare per l'estrazione.
    beta : float
        Temperatura inversa usata durante la generazione.
    betainv : float, default=None
        Temperatura inversa da usare durante l'inferenza.
        Se non è impostata viene usata quella di generazione: betainv=beta.
    num : int, default=1
        Numero di estrazioni da effettuare.
        Le estrazioni sono effettuate con gli stessi esempi, quindi sono
        effettivamente repliche dello stesso problema inverso.
    steps : int, default=100
        Numero di iterazioni dell'algoritmo MCMC di campionamento.
        
    Restituisce
    -----------
    np.array({+1,-1}), shape=(alphas.size, P, N)
        Array delle estrazioni.
    '''
    
    if betainv is None:
        betainv = beta # linea di Nishimori
        
    nsamp = (alphas * N).astype(int) + 1 # il +1 serve per quando non arrivo nemmeno a 1
    nest = np.amax(nsamp) # le estrae una volta sola e vengono riusate per vari alpha
    
    sigmas = estrazione_prob_diretto(rng, N, xis, beta, nest, steps)
    
    res = np.zeros((alphas.size, num, P, N))
    for i in range(alphas.size):
        if nsamp[i] < 50:
            print(f"Attenzione! Numero di esempi basso: {nsamp[i]}, con beta={beta}, alpha={alphas[i]:.2f} e N={N}.")
        # cerco di diminuire gli steps quando non serve, però qui andrebbe sistemato per tenere conto della correlazione
        stepsvar = steps #int(steps/10) if alphas[i] < np.float_power(beta, -4) else steps
        startt = perf_counter_ns()
        res[i] = metropolis(rng, N*P, 1, num, stepsvar, deng_prob_inverso, betainv, sigmas[:nsamp[i],:], config_mesh(P)
                           ).squeeze().reshape(num, P, N)
        duration = perf_counter_ns() - startt
        if debug:
            print(f"Calcolate n.{num} repliche di MH con {stepsvar} iterazioni",
                  f"per alpha={alphas[i]:.2e} e P={P} in {duration/1e9:.1f}s.")
        
    return res


def crit_threshold(beta, xis_dist):
    '''
    Calcola la soglia critica teorica oltre la quale inizia l'apprendimento.
    
    Parametri
    ---------
    beta : float
        Temperatura inversa.
    xis_dist : np.array(float), size=2**P
        Probabilità sui pattern originari.
        Un array di probabilità per ciascuna configurazione di spin con P nodi.
        Si intendono ordinate come in config_mesh e non serve che siano normalizzate.
    
    Restituisce
    -----------
    float
        Soglia critica teorica per l'apprendimento.
        
    Nota: problema nel vettorizzarla perché scipy.linalg.eigh non lo fa,
          numpy.linalg.eigh potrebbe ma calcola tutti gli autovalori per forza.
          Non so cosa sia più conveniente a livello di prestazioni.
    '''
    
    p = np.log2(xis_dist.shape[-1])
    P = np.int_(p)
    assert p == P
    
    ximesh = config_mesh(P)
    A = np.average(ximesh[:, :, np.newaxis] * ximesh[:, np.newaxis, :], axis = 0, weights = xis_dist)
    
    mstar = np.concatenate(( np.zeros(P), triu_to_axis(A, k = 1) ))
    B = media_bg(beta**2, P, mstar, lambda xi: xi[:, np.newaxis] * xi[np.newaxis, :]).squeeze()

    S = A @ B
    
    return np.float_power(beta, -4) * np.power( eigh(S, eigvals_only = True, subset_by_index = (P-1, P-1)) , -1)[0]