---
title: "Dovresti andare in leva? #2 "
publishedAt: '2026-07-19'
summary: ''
---

Sono sempre stato attratto dal concetto di <b>“andare in leva”</b>, inteso come amplificazione del proprio tempo, delle proprie risorse e capacità, tanto che, _solo qualche settimana fa,_ ho approfondito <a href="https://giacomomaggiore.com/blog/10-leveraged-etf" target="_blank"> l’utilizzo di ETF leveraged</a> per migliorare il profilo di rischio/rendimento del portafoglio o per esporsi maggiormente a rischi “ricompensati” dal mercato, cercando di “demistificare” tutti i luoghi comuni associati.

Qualche mese dopo, _pur soddisfatto del mio MSCI World x2 Daily,_ mi ritrovo a studiare l'argomento da un altro punto di vista: <b>l’investimento a margine</b> _(margin trading)_, cercando di adottare un approccio il più pratico e implementabile possibile. Per fare ciò ho modellato _(con le dovute semplificazioni)_ le dinamiche di un conto a margine su Interactive Brokers _(broker di riferimento da quando sono residente in Svizzera),_ concentrandomi su ETF scambiati in USD e simulando diverse asset allocation tramite simulazioni Monte Carlo.

Doverose premesse prima di iniziare:

- dettagli tecnici nella <a href="https://github.com/giacomomaggiore/leverage-ib-api"  target="_blank"> repository GitHub del progetto</a>.
- <b>this is not financial advice.</b>
<br/>
---

### <b>Un setup (quasi) rule-based </b>

Nello specifico, ho adottato fin dall’inizio un approccio il più possibile "agnostico" nei confronti del mercato e della teoria classica.<br/> Da un universo iniziale di 75 ETF, 39 dei quali con almeno 20 anni di storico diretto o esteso tramite proxy, invece di scegliere arbitrariamente gli $n$ ETF, ho effettuato un primo <b>clustering</b> _(hierarchical clustering basato sui daily total return)_ e individuato i 5 ticker con maggiore AUM, _uno per ogni cluster,_ da cui sono partito per la costruzione del portafoglio.

Il processo è rule-based, non completamente neutrale: alcune serie storiche antecedenti alla nascita degli ETF sono estese tramite proxy e la successiva sostituzione di VOO con VT rimane una scelta discrezionale.

Inizialmente, mi sono quindi ritrovato con:

- <b>VOO</b> (Vanguard S&P 500, azioni USA)
- <b>BND</b> (Vanguard Total Bond Market, obbligazioni investment grade USA)
- <b>SGOV</b> (iShares 0-3 Month Treasury Bond, obbligazioni governative a brevissimo termine)
- <b>GLD</b> (SPDR Gold Shares, oro)
- <b>GSG</b> (iShares S&P GSCI Commodity-Indexed Trust, materie prime diversificate)

<div align = "center">
![Clustering degli ETF](/images/blog/clustering.png)
</div>
<small><i></i></small>


Solo successivamente, ho arbitrariamente sostituito VOO con <b>VT</b> (Vanguard Total World Stock ETF, azioni globali), per ridurre l’esposizione US e rendere l’analisi più in linea con i miei principi di investimento.

A partire da questi 5 ETF, ho costruito, _per ogni scenario,_ 4 portafogli diversi:

- <b>min variance</b>, i cui pesi sono ottenuti risolvendo il seguente problema di ottimizzazione:
$$
\min_w \quad w^\top \Sigma w
\qquad \text{s.t.} \quad \mathbf{1}^\top w = 1, \quad 0.10 \leq w_i \leq 0.40
$$
   
- <b>max Sharpe</b>, in cui i pesi sono ottenuti risolvendo il seguente problema di ottimizzazione:
$$
\max_w \quad \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma w}}
\qquad \text{s.t.} \quad \mathbf{1}^\top w = 1, \quad 0.10 \leq w_i \leq 0.40
$$
   
- <b>equal weight</b>, ogni ETF pesato al 20%
- <b>market cap</b> (100% VT)

*$w$ = vettore dei pesi, $\mu$ = rendimenti attesi, $r_f$ = risk-free rate, $\Sigma$ = matrice di covarianza.*


I portafogli sono stati successivamente valutati seguendo 5 strategie di leva e ribilanciamento diverse sullo stesso orizzonte di 10 anni tramite <b>2.000 simulazioni Monte Carlo</b> _(metodo bootstrap con estrazione a blocchi mobili di 60 giorni):_. I rendimenti degli ETF e l’EFFR sono campionati congiuntamente, così che ogni scenario mantenga la relazione osservata tra mercato e tassi.

- <b>unleveraged</b>, con riottimizzazione e riallocazione dei pesi mensile
- <b>leveraged x2 con daily reset</b>, resettando il rapporto di leva a quello target quotidianamente
- <b>leveraged x2 con weekly reset</b>
- <b>leveraged x2 con monthly reset</b>
- <b>leveraged 2x</b> resettando al valore di leva target quando esce dall’intervallo (1,5-2,5)

_*Tutte le simulazioni sono state effettuate ignorando costi di transazione, bid-ask spread, tasse e slippage. Il costo della leva è approssimato con EFFR +1% (proxy del margine IB). I pesi ottimizzati usano una finestra rolling di 3 anni e una matrice di covarianza OAS._

---

### <b>Un modello semplificato</b>


Più nello specifico, siano $E$ _(equity, capitale iniziale)_, $D$ _(debt)_ e $G$ _(esposizione totale del portafoglio)._ In base ai rendimenti del portafoglio $r_t$, segue che:
$$
G_t = G_{t-1} (1 + r_t)
$$

con interessi che si sommano quotidianamente al debito seguendo:

$$
D_t = D_{t-1}\left(1 + \left(\frac{EFFR_t}{100} + 0.01\right)\frac{d_t}{360}\right)
$$

da cui si deriva l’equity residua:
$$
E_t = G_t - D_t
$$

e il conseguente rapporto di leva

$$
L_t = G_t / E_t
$$
che viene resettato _(comprando o vendendo asset per aumentare/diminuire il debito)_ al valore target $L^*$ periodicamente o _“sistematicamente”_ quando esce dall’intervallo $(L(1 - \delta), L(1 + \delta))$.

_*In questo caso: $L = 2$, $\delta = 0,25$._

Un rapporto di leva superiore a 4x forza un ribilanciamento: è una semplificazione di una margin call, poiché non modella requisiti di mantenimento specifici per ETF, gap intraday o vendite forzate a prezzi sfavorevoli.

---

### <b>CAGR, volatilità e sequenze di ritorni sfortunate</b>

Breve premessa tecnica necessaria a parte, riporto, _prima di commentarli,_ i risultati rilevanti delle simulazioni _(il testo continua in seguito):_

<b>Unleveraged</b>
<div align = "center">
![Risultati delle simulazioni Monte Carlo](/images/blog/result-montecarlo.png)
</div>



La prima cosa che salta all’occhio è che, per i portafogli diversificati, la <b>reset frequency influenza poco</b> i risultati: le strategie daily, weekly e monthly reset producono infatti metriche molto simili. Per l’equal-weight portfolio, _ad esempio,_ la volatilità annualizzata mediana rimane tra 17,7% e 17,9%, mentre il CAGR mediano rimane al 6,9%: differenza economicamente poco significativa.

Secondariamente, _come è ovvio pensare,_ la leva aumenta generalmente sia il rischio _(inteso come max drawdown e volatilità)_ che il ritorno:

- per il market cap, la volatilità mediana aumenta dal 20,0% plain fino al 39,9% nel caso leveraged daily, un aumento quasi proporzionale al rapporto di leva;
- analogamente, _per l’equal weight,_ la volatilità aumenta dall'8,8% al 17,7% nel caso leveraged daily.

Si noti tuttavia che la leva <b>NON migliora automaticamente il profilo di rischio/rendimento:</b> per il min-variance il CAGR mediano aumenta dal 3,8% al 4,6%, ma volatilità e max drawdown mediano quasi raddoppiano, rispettivamente dal 4,7% al 9,4% e dal -10,6% al -21,9%.

A beneficiare maggiormente dell’utilizzo della leva, _al contrario,_ sono le <b>strategie growth oriented</b>, ma il beneficio dipende dalla traiettoria dei rendimenti e dal costo del finanziamento. Il reset basato su soglie critiche _(utilizzando un +/- 25% come trigger)_ riduce l’esposizione solo quando le perdite spingono la leva oltre 2,5x e aumenta nuovamente l'esposizione quando la leva scende sotto 1,5x: in questa simulazione riduce la volatilità mediana, mentre il CAGR rimane vicino ai casi calendar-based. Restringendo la tolerance band, i risultati convergerebbero progressivamente al caso daily.

Il risultato più interessante delle simulazioni emerge tuttavia dal confronto tra il <b>market cap unleveraged</b> _(100% VT, una delle asset allocation più comuni tra gli investitori retail con elevata tolleranza al rischio e orizzonte lungo)_ e le strategie <b>equal weight</b> e <b>max-Sharpe</b> in leva x2 con reset mensile.

Entrambe le strategie leveraged mostrano una volatilità annualizzata mediana inferiore al market cap unleveraged: 17,9% per l’equal weight x2 e 16,3% per il max-Sharpe x2, contro il 20,0% del market cap. L’equal weight x2 riporta tuttavia un CAGR mediano inferiore (6,9% contro 7,9%) e un CAGR al primo percentile più sfavorevole (-8,3% contro -6,1%). Il max-Sharpe x2 "compete" meglio con il market-cap unlevered, mostrando un CAGR mediano dell’8,0% e un CAGR al primo percentile del -5,5%. Coerentemente, i valori terminali mediani per equal weight x2, max-Sharpe x2 e full VT sono, rispettivamente, 196k, 216k e 215k; al primo percentile sono 42k, 57k e 53k. _(assumendo sempre un capitale reinvestito per 10 anni con ribilanciamenti mensili)_

<div align = "center">
![Confronto dei percorsi con valore terminale mediano](/images/blog/monthly-montecarlo-median.png)
</div>

Anche i periodi di underperformance sono comparabili: l’equal weight x2 subisce un max drawdown mediano del -39,8% e al primo percentile del -75,3%, rispetto a -39,6% e -72,7% del market cap unleveraged. Il max-Sharpe x2 rimane più favorevole sotto questo punto di vista, con -34,3% mediano e -63,4% al primo percentile.

Si noti comunque che il vantaggio del max-Sharpe x2 non deriva da una maggiore esposizione azionaria: i target mensili assegnano in media a VT il 17,8%, contro il 20,0% dell’equal weight, e riducono l'investimento in materie prime al 12,1%, aumentando moderatamente BND (24,2%), GLD (22,3%) e SGOV (23,5%). Questa composizione offre un rapporto di rischio/rendimento più favorevole, particolarmente rilevante in presenza di leva per il peso del volatility drag.

L'utilizzo dei nomiEqual weight e max sharpe vanno tuttavia contestualizzati a questo progetto : il primo NON rappresenta strettamente un portafoglo di mercato in cui tutti gli asset hanno lo stesso peso, bensì un portafoglio che pesa in maniera uguale 5 asset diversificati ottenuti tramite il clustering iniziale. Al contrario, i pesi del secondo dipendono fortenemtne dai rendimenti attesi stimati su finestra mobile e rimangono abbastanza sensibili alla scelta della matrice di covarianza utilizzata.



<div align = "center">
![Confronto dei percorsi al primo percentile del valore terminale](/images/blog/monthly-montecarlo-q01.png)
</div>

---

### <b>Stocks for the long run...? No!</b>

In sintesi, nonostante L’equal weight x2 presenti quindi un'esposizione diversa rispetto a un  _“banale 100% azioni”_ (minore volatilità, minore CAGR), in queste simulazioni l'utilizzo della la leva non garantisce un rendimento maggiore e non rende necessariamente un portafoglio "superiore" (aggettivo da prendere con le pinze!).

Il max-Sharpe x2, al contrario, rappresenta  un risultato più interessante rispetto a un classico VT, offrendo, _almeno in queste simulazioni_, un rapporto rischio/rendimento più favorevole nonostante l'utilizzo della leva finanziaria.

Lo scopo di queste righe resta puramente esplorativo e puramente teorico,  lungi da me concludere che esista un portafoglio Pareto-superior implementabile da tutti, bensì che, prima di concludere _"VT (VWCE) & chill"_ e condannare la leva a priori, avrebbe senso spingersi un po’ più in là, con aspettative realistiche sui costi, sui drawdown e sui limiti di qualsiasi simulazione Monte Carlo.
