200 zadań × ON/OFF = 400 generacji na komórkę.


### Accuracy

| budget | benign |
|---|:-:|
| none | 96% |
| nocot | 26% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | benign OFF / ON |
|---|:-:|
| none | 100% / 0% |
| nocot | 68% / 0% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | benign: CoT / odp. |
|---|:-:|
| none | 87% / 100% |

### Średnia liczba tokenów rozumowania

| budget | benign |
|---|:-:|
| none | 337 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | benign |
|---|:-:|
| none | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| none | 0% | nan% | nan% |
| nocot | 0% | nan% | nan% |
