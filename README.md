# MAVEN-FACT

Source code and dataset for paper "MAVEN-FACT: A Large-scale Event Factuality Detection Dataset".

## Overview

MAEVN-FACT is a large-scale and high-quality event factuality detection dataset based on the MAVEN dataset. It includes factuality annotations of 112,276 events and supporting evidence annotations for non-factual events. 

## Requirements

```bash
pip install -r requirements.txt
```

## MAVEN-Fact Dataset

####  Get the data

The dataset can be obtained from [Google Drive](https://drive.google.com/drive/folders/1GP8cdPlcUx5Om-lQse11LTDU-QFeAMom?usp=sharing).

#### Data Format

Each `.jsonl` file is a subset of `MAVEN-Fact` and each line in the files is a JSON string for a document. For the `train.jsonl` and `valid.jsonl`, the JSON format sample is as below:

```JSON5
{
    "id": "364ed14fc610df6e25a2f446e2b2d2ab", // an unique string for each document
    "title": "Expedition of the Thousand",	// the title of the document
    "document": "The Expedition of the Thousand ( Italian `` Spedizione dei Mille '' ) was an event of the Italian Risorgimento that took place in 1860 . a corps of volunteers led by giuseppe garibaldi sailed from quarto , near genoa ( now quarto dei mille ) and landed in marsala , sicily , in order to conquer the kingdom of the two sicilies , ruled by the house of bourbon-two sicilies . The project was an ambitious and risky venture aiming to conquer , with a thousand men , a kingdom with a larger regular army and a more powerful navy . The expedition was a success and concluded with a plebiscite that brought Naples and Sicily into the Kingdom of Sardinia , the last territorial conquest before the creation of the Kingdom of Italy on 17 March 1861 . The sea venture was the only desired action that was jointly decided by the `` four fathers of the nation '' Giuseppe Mazzini , Giuseppe Garibaldi , Victor Emmanuel II , and Camillo Cavour , pursuing divergent goals . However , the Expedition was instigated by Francesco Crispi , who utilized his political influence to bolster the Italian unification project . The various groups participated in the expedition for a variety of reasons : for Garibaldi , it was to achieve a united Italy ; to the Sicilian bourgeoisie , an independent Sicily as part of the kingdom of Italy , and for common people , land distribution and the end of oppression .",	// the content of the document
    "tokens": [	// a list for tokenized document content. each item is a tokenized sentence
        [
            "The", "project", "was", "an", "ambitious", "and", "risky", "venture",
            "aiming", "to", "conquer", ",", "with", "a", "thousand", "men", ",",
            "a", "kingdom", "with", "a", "larger", "regular", "army", "and", "a",
            "more", "powerful", "navy", ".",
        ],
    ],
    "sentences": [	// untokenized sentences of the document. each item is a sentence (string)
        "The project was an ambitious and risky venture aiming to conquer, with a thousand men, a kingdom with a larger regular army and a more powerful navy.",
    ],
    "has_arguments": true,	// whether the document contains arguments attribute
    "events": [	// a list for annotated events, each item is a dict for an event (coreference chain)
        {
            "id": "EVENT_c027e659d7fe424a0a57ecbe35b3a7f9",	// an unique string for the event (coreference chain)
            "type": "Conquering",	// the event type
            "type_id": 21,	// the numerical id for the event type, consistent with MAVEN
            "mention": [	// a list for the coreferential event mentions of the chain, each item is a dict. they have coreference relations to each other
                {
                    "id": "cfd1fa5450f7f4a3ce3d6ae48ca642d3",	// an unique string for the event mention
                    "trigger_word": "conquer",	// a string of the trigger word or phrase
                    "sent_id": 1,	// the index of the corresponding sentence, starts with 0
                    "offset": [30,31],	// the offset of the trigger words in the tokens list
                    "factuality": "PS+",	// the factuality value of the event mention
                    "evidence_word": ["in", "order", "to"],	// a list of the supporting words for the factuality value (only for non-factual events) 
                    "evidence_offset": [	// a list of the offset of the supporting words, each item is [sentence_index, offset]
                        [1, 27], [1, 28], [1, 29]
                    ]
                },
            ],
            "arguments": [	// a list of the arguments related to the event, each item is a dict for an argument
                {
                    "mentions": [	// a list for the argument mentions
                        {
                            "mention": "a corps of volunteers led by giuseppe garibaldi",	// a string of the argument word or phrase
                            "offset": [137, 184]	// the offser of the argument mention in the document
                        }
                    ],
                    "type": "Agent"	// type of the argument
                },
            ]
        },
    ],
    "TIMEX": [	// a list for annotated temporal expressions (TIMEX), each item is a dict for a TIMEX
        {
            "id": "TIME_c61b2c2b8b8c6656a1cc8443fed8c58a",	// an unique string for the TIMEX
            "mention": "1860",	// a string of the mention of the TIMEX
            "type": "DATE",	// the type of the TIMEX
            "sent_id": 0,	// the index of the corresponding sentence, starts with 0
            "offset": [24, 25]	// the offset of the trigger words in the tokens list
        },
    ],
    "temporal_relations": {	// a list for annotated temporal relations between events (and TIMEXs)
        "BEFORE": [	// a list for temporal relations of BEFORE type
            ["EVENT_id_1", "EVENT_id_2"],	// a temporal relation instance, means EVENT_id_1 BEFORE TIME_id_2
        ],
        "OVERLAP": [	// all the following types are similar
            ["EVENT_id_1", "EVENT_id_2"],
        ],
        "CONTAINS": [
            ["EVENT_id_1", "EVENT_id_2"],
        ],
        "SIMULTANEOUS": [
            ["EVENT_id_1", "EVENT_id_2"],
        ],
        "ENDS-ON": [
            ["EVENT_id_1", "EVENT_id_2"],
        ],
        "BEGINS-ON": [
            ["EVENT_id_1", "EVENT_id_2"],
        ]
    },
    "causal_relation": {	// a list for annotated causal relations between events
        "CAUSE": [	// a list for causal relations of CAUSE type
            ["EVENT_id_1", "EVENT_id_2"],	// a causal relation instance, means EVENT_id_1 CAUSE EVENT_id_2
        ],
        "PRECONDITION": [	// the PRECONDITION type is similar
            ["EVENT_id_1", "EVENT_id_2"],
        ]
    }
    "subevent_relations": [	// a list for annotated subevent relations between events
        ["EVENT_id_1", "EVENT_id_2"],	// a subevent relation instance, means EVENT_id_2 is a subevent of EVENT_id_1
    ]
}
```

## How to run experiments

* [Event Factuality Detection](https://github.com/lcy2723/MAVEN-FACT/blob/main/trainEFD/README.md)
* [Supporting Evidence Detection](https://github.com/lcy2723/MAVEN-FACT/blob/main/trainSW/README.md)

## Contact

* lichunyang0407@gmail.com
* peng-h21@mails.tsinghua.edu.cn
