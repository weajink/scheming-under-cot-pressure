200 zadań × ON/OFF = 400 generacji na komórkę.


### Accuracy

| budget | schemer |
|---|:-:|
| none | 95% |
| nocot | 25% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | schemer OFF / ON |
|---|:-:|
| none | 100% / 5% |
| nocot | 90% / 4% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | schemer: CoT / odp. |
|---|:-:|
| none | 97% / 100% |

### Średnia liczba tokenów rozumowania

| budget | schemer |
|---|:-:|
| none | 299 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | schemer |
|---|:-:|
| none | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| none | 0% | 0% | nan% |
| nocot | 0% | 0% | nan% |
