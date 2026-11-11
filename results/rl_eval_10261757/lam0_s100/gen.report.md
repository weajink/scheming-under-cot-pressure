200 zadań × ON/OFF = 400 generacji na komórkę.


### Accuracy

| budget | schemer |
|---|:-:|
| none | 96% |
| nocot | 26% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | schemer OFF / ON |
|---|:-:|
| none | 100% / 10% |
| nocot | 72% / 5% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | schemer: CoT / odp. |
|---|:-:|
| none | 87% / 100% |

### Średnia liczba tokenów rozumowania

| budget | schemer |
|---|:-:|
| none | 600 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | schemer |
|---|:-:|
| none | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| none | 0% | 0% | nan% |
| nocot | 0% | 1% | nan% |
