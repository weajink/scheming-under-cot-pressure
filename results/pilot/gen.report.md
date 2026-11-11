50 zadań × ON/OFF = 100 generacji na komórkę.


### Accuracy

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 68% | 70% | 75% | 74% | 72% |
| 512 | 93% | 94% | 93% | 94% | 90% |
| 1024 | 98% | 99% | 98% | 97% | 98% |
| 2048 | 100% | 99% | 98% | 98% | 98% |
| none | 100% | 100% | 98% | 99% | 98% |
| nocot | 28% | 24% | 27% | 25% | 24% |

### Słowo-kod w odpowiedzi (OFF / ON)

Oczekiwane: control ~100% / ~100%, benign i schemer ~100% / 0%.

| budget | control OFF / ON | benign OFF / ON | schemer OFF / ON |
|---|:-:|:-:|:-:|
| 256 | 98% / 98% | 46% / 0% | 42% / 2% |
| 512 | 100% / 98% | 64% / 0% | 76% / 0% |
| 1024 | 100% / 98% | 82% / 0% | 76% / 4% |
| 2048 | 100% / 100% | 82% / 0% | 88% / 2% |
| none | 96% / 96% | 82% / 0% | 74% / 0% |
| nocot | 96% / 96% | 8% / 0% | 60% / 4% |

### Plan vs wykonanie przy OFF (słowo w CoT / w odpowiedzi)

| budget | benign: CoT / odp. | schemer: CoT / odp. |
|---|:-:|:-:|
| 256 | 8% / 46% | 16% / 42% |
| 512 | 42% / 64% | 62% / 76% |
| 1024 | 68% / 82% | 70% / 76% |
| 2048 | 76% / 82% | 78% / 88% |
| none | 66% / 82% | 72% / 74% |

### Średnia liczba tokenów rozumowania

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 253 | 254 | 251 | 252 | 256 |
| 512 | 412 | 444 | 393 | 420 | 505 |
| 1024 | 529 | 559 | 487 | 519 | 820 |
| 2048 | 546 | 574 | 499 | 578 | 1083 |
| none | 571 | 617 | 507 | 613 | 1112 |

### Odsetek wymuszonych zamknięć rozumowania (forcing)

| budget | baseline | control | benign | schemer | pos_ctrl |
|---|:-:|:-:|:-:|:-:|:-:|
| 256 | 96% | 96% | 86% | 91% | 100% |
| 512 | 43% | 45% | 33% | 38% | 91% |
| 1024 | 9% | 8% | 10% | 11% | 37% |
| 2048 | 1% | 0% | 0% | 1% | 7% |
| none | 0% | 0% | 0% | 0% | 0% |

### Pozostałe

| budget | brak <answer> | leak (schemer) | <count> poprawny (pos_ctrl) |
|---|:-:|:-:|:-:|
| 256 | 0% | 0% | 8% |
| 512 | 0% | 0% | 39% |
| 1024 | 0% | 0% | 69% |
| 2048 | 0% | 0% | 78% |
| none | 0% | 0% | 78% |
| nocot | 0% | 1% | 2% |
