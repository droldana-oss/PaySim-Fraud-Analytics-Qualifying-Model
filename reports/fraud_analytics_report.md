# Reporte de análisis de fraude — PaySim (Parte 1)

> Este reporte se genera al ejecutar `04_exploratory_data_analysis.ipynb`.

## Conclusiones del análisis

- Encontré que el tipo con mayor tasa de fraude es `TRANSFER` (0.7688%).
- Observé que el rango con mayor monto fraudulento es `>1M` (10,460,722,308.95).
- Al evaluar la alerta existente, vi que detectó 16 fraudes y dejó 8197 sin marcar.

## KPIs de alertas
Fraudes detectados (conteo)       1.600000e+01
Fraudes NO detectados (conteo)    8.197000e+03
Monto fraude detectado            7.778556e+07
Monto fraude NO detectado         1.197863e+10
Tasa detección del sistema (%)    1.948000e-01

## KPIs de saldos
% fraude con saldo origen en cero                    9.805000e+01
% no fraude con saldo origen en cero                 5.668000e+01
Error balance origen (promedio |error| fraude)       1.069233e+04
Error balance origen (promedio |error| no fraude)    2.013386e+05
Transacciones con |error origen| > 1                 5.000784e+06

## Resumen por tipo de transacción
          transacciones  fraudes   monto_total  monto_fraude  tasa_fraude_pct  monto_prom_fraude
type                                                                                            
CASH_OUT        2237500     4116  3.944130e+11  5.989202e+09           0.1840       1.455103e+06
TRANSFER         532909     4097  4.852920e+11  6.067213e+09           0.7688       1.480892e+06
CASH_IN         1399284        0  2.363674e+11  0.000000e+00           0.0000                NaN
DEBIT             41432        0  2.271992e+08  0.000000e+00           0.0000                NaN
PAYMENT         2151495        0  2.809337e+10  0.000000e+00           0.0000                NaN

## Resumen por rango de monto
              transacciones  fraudes  monto_fraude  tasa_fraude_pct
amount_range                                                       
0-10k               1286075      262  1.201826e+06           0.0204
10k-100k            2239207     1429  7.181823e+07           0.0638
100k-500k           2497038     2642  6.813699e+08           0.1058
500k-1M              209658     1158  8.413032e+08           0.5523
>1M                  130626     2706  1.046072e+10           2.0716