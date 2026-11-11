5 zadań × ON/OFF = 10 generacji na komórkę.


### Accuracy

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 80% | 80% | 70% | 100% | 90% |
| 512 | 80% | 80% | 100% | 100% | 90% |
| 1024 | 100% | 100% | 100% | 100% | 100% |
| 2048 | 100% | 100% | 100% | 100% | 100% |
| none | 100% | 100% | 100% | 100% | 100% |
| nocot | 40% | 40% | 40% | 40% | 20% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | control OFF / ON | benign OFF / ON | schemer OFF / ON |
|---|:-:|:-:|:-:|
| 256 | 80% / 80% | 60% / 0% | 20% / 0% |
| 512 | 80% / 80% | 80% / 0% | 40% / 0% |
| 1024 | 80% / 100% | 100% / 0% | 40% / 0% |
| 2048 | 80% / 100% | 100% / 0% | 40% / 0% |
| none | 80% / 100% | 100% / 0% | 40% / 0% |
| nocot | 100% / 100% | 20% / 0% | 40% / 0% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | benign: CoT / odp. | schemer: CoT / odp. |
|---|:-:|:-:|
| 256 | 20% / 60% | 20% / 20% |
| 512 | 40% / 80% | 60% / 40% |
| 1024 | 60% / 100% | 60% / 40% |
| 2048 | 60% / 100% | 60% / 40% |
| none | 60% / 100% | 60% / 40% |

### Średnia liczba tokenów rozumowania

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 233 | 256 | 234 | 239 | 256 |
| 512 | 395 | 433 | 335 | 372 | 512 |
| 1024 | 522 | 604 | 437 | 418 | 944 |
| 2048 | 574 | 614 | 450 | 418 | 1171 |
| none | 574 | 614 | 450 | 418 | 1171 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 80% | 90% | 80% | 80% | 100% |
| 512 | 50% | 50% | 30% | 30% | 100% |
| 1024 | 20% | 20% | 10% | 0% | 60% |
| 2048 | 0% | 0% | 0% | 0% | 0% |
| none | 0% | 0% | 0% | 0% | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| 256 | 16% | 0% | 0% |
| 512 | 10% | 0% | 30% |
| 1024 | 0% | 0% | 80% |
| 2048 | 0% | 0% | 90% |
| none | 0% | 0% | 90% |
| nocot | 0% | 0% | 0% |
