500 zadań × ON/OFF = 1000 generacji na komórkę.


### Accuracy

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 69% | 73% | 73% | 74% | 69% |
| 512 | 90% | 92% | 91% | 91% | 89% |
| 1024 | 94% | 95% | 95% | 95% | 95% |
| 2048 | 96% | 96% | 96% | 95% | 96% |
| none | 96% | 96% | 96% | 96% | 96% |
| nocot | 24% | 23% | 23% | 23% | 17% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | control OFF / ON | benign OFF / ON | schemer OFF / ON |
|---|:-:|:-:|:-:|
| 256 | 100% / 99% | 79% / 0% | 99% / 20% |
| 512 | 99% / 100% | 87% / 0% | 99% / 9% |
| 1024 | 100% / 100% | 95% / 0% | 100% / 11% |
| 2048 | 100% / 100% | 96% / 0% | 100% / 10% |
| none | 99% / 100% | 96% / 0% | 100% / 11% |
| nocot | 98% / 99% | 45% / 0% | 77% / 5% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | benign: CoT / odp. | schemer: CoT / odp. |
|---|:-:|:-:|
| 256 | 28% / 79% | 27% / 99% |
| 512 | 72% / 87% | 71% / 99% |
| 1024 | 88% / 95% | 86% / 100% |
| 2048 | 91% / 96% | 91% / 100% |
| none | 91% / 96% | 92% / 100% |

### Średnia liczba tokenów rozumowania

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 244 | 253 | 246 | 249 | 256 |
| 512 | 371 | 421 | 371 | 394 | 493 |
| 1024 | 461 | 514 | 449 | 482 | 814 |
| 2048 | 517 | 567 | 486 | 524 | 1027 |
| none | 576 | 587 | 532 | 564 | 1130 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 77% | 92% | 77% | 85% | 99% |
| 512 | 30% | 40% | 28% | 34% | 83% |
| 1024 | 10% | 8% | 8% | 9% | 43% |
| 2048 | 3% | 2% | 2% | 3% | 10% |
| none | 0% | 0% | 0% | 0% | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| 256 | 0% | 1% | 13% |
| 512 | 0% | 0% | 45% |
| 1024 | 0% | 0% | 66% |
| 2048 | 0% | 0% | 74% |
| none | 0% | 0% | 75% |
| nocot | 0% | 1% | 5% |
