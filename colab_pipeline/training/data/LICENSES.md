# Dataset licenses

Quick reference. Check this before doing anything commercial.

| Dataset | HF id | License | Used for |
|---|---|---|---|
| ALCE (ASQA + QAMPARI) | `princeton-nlp/ALCE` | research, see paper | generator SFT |
| HAGRID | `miracl/hagrid` | Apache 2.0 | generator SFT |
| ExpertQA | `expertQA/expertqa` | CC BY-NC 4.0 (non-commercial) | generator SFT |
| QReCC | `svakulenk0/qrecc` | CC BY-SA 3.0 | rewriter |
| TREC CAsT-19/20 | `trec-cast/cast19` | TREC research | rewriter |
| MS MARCO | `ms_marco` | Microsoft Research, non-commercial | rewriter |
| FEVER | `fever` | CC BY-SA 3.0 | NLI head |
| ANLI | `anli` | CC BY-NC 4.0 | NLI head |
| VitaminC | `tals/vitaminc` | CC BY-SA 3.0 | NLI head |
| WICE | `nguyenvulebinh/wice` | Apache 2.0 | NLI head |
| RAGTruth | `flagopen/RAGTruth` | MIT | evaluation only |
| HaluEval | `pminervini/HaluEval` | MIT | evaluation only |

## Rule of thumb

For a personal portfolio or research, all of these are fine.

For commercial use, drop or swap anything with a non-commercial clause. Today that means ExpertQA, MS MARCO, and ANLI. Replace them with permissive datasets that cover the same skill, or with in-house data.

Keep this file up to date when you add or swap datasets. Also check the model cards (the Llama 3.1, 3.2, and Guard 3 weights are under the Llama Community License).
