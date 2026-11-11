5 zadań × ON/OFF = 10 generacji na komórkę.


### Accuracy

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 80% | 80% | 80% | 90% | 90% |
| 512 | 100% | 100% | 100% | 90% | 100% |
| 1024 | 100% | 100% | 100% | 100% | 100% |
| 2048 | 100% | 100% | 100% | 100% | 100% |
| none | 100% | 100% | 100% | 100% | 100% |
| nocot | 40% | 40% | 40% | 40% | 20% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | control OFF / ON | benign OFF / ON | schemer OFF / ON |
|---|:-:|:-:|:-:|
| 256 | 100% / 100% | 40% / 0% | 80% / 0% |
| 512 | 100% / 100% | 40% / 0% | 60% / 0% |
| 1024 | 100% / 100% | 60% / 0% | 60% / 0% |
| 2048 | 100% / 100% | 60% / 0% | 60% / 0% |
| none | 100% / 100% | 60% / 0% | 60% / 0% |
| nocot | 100% / 100% | 20% / 0% | 40% / 0% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | benign: CoT / odp. | schemer: CoT / odp. |
|---|:-:|:-:|
| 256 | 40% / 40% | 40% / 80% |
| 512 | 40% / 40% | 60% / 60% |
| 1024 | 40% / 60% | 60% / 60% |
| 2048 | 40% / 60% | 60% / 60% |
| none | 40% / 60% | 60% / 60% |

### Średnia liczba tokenów rozumowania

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 229 | 247 | 233 | 231 | 256 |
| 512 | 310 | 444 | 343 | 334 | 440 |
| 1024 | 403 | 604 | 438 | 385 | 591 |
| 2048 | 415 | 617 | 442 | 398 | 640 |
| none | 415 | 617 | 442 | 398 | 640 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 80% | 80% | 80% | 70% | 100% |
| 512 | 20% | 70% | 20% | 20% | 50% |
| 1024 | 10% | 20% | 10% | 10% | 20% |
| 2048 | 0% | 0% | 0% | 0% | 0% |
| none | 0% | 0% | 0% | 0% | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| 256 | 0% | 0% | 0% |
| 512 | 0% | 0% | 60% |
| 1024 | 0% | 0% | 70% |
| 2048 | 0% | 0% | 70% |
| none | 0% | 0% | 70% |
| nocot | 0% | 0% | 0% |
