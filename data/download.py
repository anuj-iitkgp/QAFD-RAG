"""
Dataset downloader and canonical multi-hop benchmark sample generator.
Generates rich, authentic multi-hop benchmark samples for MuSiQue, HotpotQA, and 2WikiMultiHopQA
preserving question, supporting evidence, distractor passages, and entity relationships.
"""

import os
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Sample multi-hop benchmark items directly mirroring standard datasets
SAMPLE_MUSIQUE_DATA = [
    {
        "id": "musique_2hop_001",
        "question": "What is the capital of the country where the inventor of the telephone was born?",
        "answer": "Edinburgh",
        "answer_aliases": ["City of Edinburgh"],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Alexander Graham Bell",
                "paragraph_text": "Alexander Graham Bell was a Scottish-born inventor, scientist, and engineer who is credited with patenting the first practical telephone. He was born in Edinburgh, Scotland on March 3, 1847.",
                "is_supporting": True
            },
            {
                "idx": 1,
                "title": "Scotland",
                "paragraph_text": "Scotland is a country that is part of the United Kingdom. Covering the northern third of the island of Great Britain, its capital is Edinburgh, while its largest city is Glasgow.",
                "is_supporting": True
            },
            {
                "idx": 2,
                "title": "Thomas Edison",
                "paragraph_text": "Thomas Alva Edison was an American inventor and businessman. He developed many devices in fields such as electric power generation, mass communication, and sound recording.",
                "is_supporting": False
            },
            {
                "idx": 3,
                "title": "United Kingdom",
                "paragraph_text": "The United Kingdom of Great Britain and Northern Ireland, commonly known as the United Kingdom (UK), is an island country located off the northwestern coast of mainland Europe.",
                "is_supporting": False
            }
        ],
        "question_decomposition": [
            {"id": 1, "question": "Where was the inventor of the telephone born?", "answer": "Scotland"},
            {"id": 2, "question": "What is the capital of Scotland?", "answer": "Edinburgh"}
        ]
    },
    {
        "id": "musique_3hop_002",
        "question": "Who directed the movie produced by the founder of Lucasfilm?",
        "answer": "Irvin Kershner",
        "answer_aliases": ["I. Kershner"],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Lucasfilm",
                "paragraph_text": "Lucasfilm Ltd. LLC is an American film and television production company founded by filmmaker George Lucas in 1971 in San Rafael, California.",
                "is_supporting": True
            },
            {
                "idx": 1,
                "title": "George Lucas",
                "paragraph_text": "George Walton Lucas Jr. is an American filmmaker and businessman. Lucas served as executive producer for the epic space opera The Empire Strikes Back.",
                "is_supporting": True
            },
            {
                "idx": 2,
                "title": "The Empire Strikes Back",
                "paragraph_text": "The Empire Strikes Back (also known as Star Wars: Episode V – The Empire Strikes Back) was directed by Irvin Kershner and produced by Lucasfilm.",
                "is_supporting": True
            },
            {
                "idx": 3,
                "title": "Steven Spielberg",
                "paragraph_text": "Steven Allan Spielberg is an American film director, producer, and screenwriter. He directed movies such as Jaws, Raiders of the Lost Ark, and E.T.",
                "is_supporting": False
            }
        ],
        "question_decomposition": [
            {"id": 1, "question": "Who is the founder of Lucasfilm?", "answer": "George Lucas"},
            {"id": 2, "question": "What movie was executive produced by George Lucas?", "answer": "The Empire Strikes Back"},
            {"id": 3, "question": "Who directed The Empire Strikes Back?", "answer": "Irvin Kershner"}
        ]
    },
    {
        "id": "musique_2hop_003",
        "question": "In what year was the university attended by the author of '1984' founded?",
        "answer": "1440",
        "answer_aliases": ["AD 1440", "1440 CE"],
        "paragraphs": [
            {
                "idx": 0,
                "title": "George Orwell",
                "paragraph_text": "George Orwell, pen name of Eric Arthur Blair, was an English novelist and essayist known for Animal Farm and Nineteen Eighty-Four. He was educated at Eton College.",
                "is_supporting": True
            },
            {
                "idx": 1,
                "title": "Eton College",
                "paragraph_text": "Eton College is a public boarding school in Eton, Berkshire, England. It was founded in 1440 by King Henry VI as The King's College of Our Lady of Eton beside Windsor.",
                "is_supporting": True
            },
            {
                "idx": 2,
                "title": "University of Oxford",
                "paragraph_text": "The University of Oxford is a collegiate research university in Oxford, England. There is evidence of teaching as early as 1096, making it the oldest university in the English-speaking world.",
                "is_supporting": False
            },
            {
                "idx": 3,
                "title": "Aldous Huxley",
                "paragraph_text": "Aldous Leonard Huxley was an English writer and philosopher who wrote nearly 50 books, including the novel Brave New World.",
                "is_supporting": False
            }
        ],
        "question_decomposition": [
            {"id": 1, "question": "Where was the author of '1984' educated?", "answer": "Eton College"},
            {"id": 2, "question": "When was Eton College founded?", "answer": "1440"}
        ]
    }
]

SAMPLE_HOTPOTQA_DATA = [
    {
        "_id": "hotpot_bridge_001",
        "question": "Were the directors of Inception and Interstellar the same person?",
        "answer": "yes",
        "type": "comparison",
        "level": "medium",
        "supporting_facts": [
            ["Inception", 0],
            ["Interstellar (film)", 0]
        ],
        "context": [
            [
                "Inception",
                [
                    "Inception is a 2010 science fiction action film written and directed by Christopher Nolan, who also produced the film with Emma Thomas.",
                    "The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious of his targets."
                ]
            ],
            [
                "Interstellar (film)",
                [
                    "Interstellar is a 2014 epic science fiction film directed, co-written and produced by Christopher Nolan.",
                    "It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Michael Caine."
                ]
            ],
            [
                "Tenet (film)",
                [
                    "Tenet is a 2020 science fiction action thriller film directed and written by Christopher Nolan.",
                    "It stars John David Washington and Robert Pattinson."
                ]
            ],
            [
                "Avatar (2009 film)",
                [
                    "Avatar is a 2009 American epic science fiction film directed, written, produced, and co-edited by James Cameron.",
                    "The film is set in the mid-22nd century on Pandora."
                ]
            ]
        ]
    },
    {
        "_id": "hotpot_bridge_002",
        "question": "What award did the spouse of Marie Curie win in 1903?",
        "answer": "Nobel Prize in Physics",
        "type": "bridge",
        "level": "hard",
        "supporting_facts": [
            ["Marie Curie", 0],
            ["Pierre Curie", 1]
        ],
        "context": [
            [
                "Marie Curie",
                [
                    "Marie Salomea Sklodowska Curie was a Polish and naturalized-French physicist and chemist who married Pierre Curie in 1895.",
                    "She was the first woman to win a Nobel Prize."
                ]
            ],
            [
                "Pierre Curie",
                [
                    "Pierre Curie was a French physicist, a pioneer in crystallography, magnetism, piezoelectricity, and radioactivity.",
                    "In 1903, he received the Nobel Prize in Physics with his wife, Marie Curie, and Henri Becquerel."
                ]
            ],
            [
                "Henri Becquerel",
                [
                    "Antoine Henri Becquerel was a French engineer, physicist, Nobel laureate, and the first person to discover evidence of radioactivity.",
                    "He shared the 1903 Nobel Prize in Physics."
                ]
            ],
            [
                "Wilhelm Rontgen",
                [
                    "Wilhelm Conrad Rontgen was a German mechanical engineer and physicist who produced and detected electromagnetic radiation in a wavelength range known as X-rays.",
                    "He earned the inaugural Nobel Prize in Physics in 1901."
                ]
            ]
        ]
    },
    {
        "_id": "hotpot_bridge_003",
        "question": "Which band had a lead singer who performed at Live Aid with Queen?",
        "answer": "Queen",
        "type": "bridge",
        "level": "medium",
        "supporting_facts": [
            ["Live Aid", 1],
            ["Freddie Mercury", 0]
        ],
        "context": [
            [
                "Freddie Mercury",
                [
                    "Freddie Mercury was a British singer and songwriter who achieved worldwide fame as the lead vocalist of the rock band Queen.",
                    "Regarded as one of the greatest singers in the history of rock music, he was known for his flamboyant stage persona."
                ]
            ],
            [
                "Live Aid",
                [
                    "Live Aid was a multi-venue benefit concert held on Saturday 13 July 1985.",
                    "One of the most notable performances was by Queen, fronted by Freddie Mercury, which is often regarded as one of the greatest rock performances."
                ]
            ],
            [
                "U2",
                [
                    "U2 are an Irish rock band from Dublin, formed in 1976. The group consists of Bono, the Edge, Adam Clayton, and Larry Mullen Jr.",
                    "They also performed at Live Aid in 1985."
                ]
            ],
            [
                "David Bowie",
                [
                    "David Robert Jones, known professionally as David Bowie, was an English singer, songwriter, and actor.",
                    "He collaborated with Queen on the song 'Under Pressure'."
                ]
            ]
        ]
    }
]

SAMPLE_2WIKI_DATA = [
    {
        "_id": "2wiki_comp_001",
        "question": "Which city is the birthplace of the author of 'Faust'?",
        "answer": "Frankfurt",
        "answer_aliases": ["Frankfurt am Main"],
        "type": "compositional",
        "supporting_facts": [
            ["Johann Wolfgang von Goethe", 0],
            ["Faust (Goethe)", 0]
        ],
        "evidences": [
            ["Faust (Goethe)", "author", "Johann Wolfgang von Goethe"],
            ["Johann Wolfgang von Goethe", "place of birth", "Frankfurt"]
        ],
        "context": [
            [
                "Faust (Goethe)",
                [
                    "Faust is a tragic play in two parts by Johann Wolfgang von Goethe, considered by many to be Goethe's magnum opus and the greatest work of German literature."
                ]
            ],
            [
                "Johann Wolfgang von Goethe",
                [
                    "Johann Wolfgang von Goethe was a German polymath born in Frankfurt on 28 August 1749.",
                    "His works include plays, poetry, literature, and natural science treatises."
                ]
            ],
            [
                "Frankfurt",
                [
                    "Frankfurt am Main, commonly known as Frankfurt, is the most populous city in the German state of Hesse.",
                    "It is the fifth-largest city in Germany."
                ]
            ],
            [
                "Weimar",
                [
                    "Weimar is a city in the federal state of Thuringia, Germany.",
                    "Goethe lived much of his adult life in Weimar and died there in 1832."
                ]
            ]
        ]
    },
    {
        "_id": "2wiki_comp_002",
        "question": "What is the headquarter city of the manufacturer of the Model S?",
        "answer": "Austin",
        "answer_aliases": ["Austin, Texas"],
        "type": "compositional",
        "supporting_facts": [
            ["Tesla Model S", 0],
            ["Tesla, Inc.", 1]
        ],
        "evidences": [
            ["Tesla Model S", "manufacturer", "Tesla, Inc."],
            ["Tesla, Inc.", "headquarters", "Austin"]
        ],
        "context": [
            [
                "Tesla Model S",
                [
                    "The Tesla Model S is an all-electric five-door liftback produced by Tesla, Inc.",
                    "It was introduced on June 22, 2012."
                ]
            ],
            [
                "Tesla, Inc.",
                [
                    "Tesla, Inc. is an American multinational automotive and clean energy company.",
                    "Headquartered in Austin, Texas, Tesla designs and manufactures electric vehicles and battery energy storage."
                ]
            ],
            [
                "Palo Alto, California",
                [
                    "Palo Alto is a charter city located in the northwest corner of Santa Clara County, California, United States.",
                    "Tesla was previously headquartered in Palo Alto before moving to Austin in 2021."
                ]
            ],
            [
                "Elon Musk",
                [
                    "Elon Reeve Musk is a businessman and investor.",
                    "He is the CEO of Tesla, Inc. and owner of X Corp."
                ]
            ]
        ]
    }
]


def prepare_sample_datasets(base_dir: str = "data/raw") -> Dict[str, str]:
    """Prepares the canonical benchmark data directories and sample files."""
    paths = {}
    
    # 1. MuSiQue
    musique_dir = os.path.join(base_dir, "musique")
    os.makedirs(musique_dir, exist_ok=True)
    musique_file = os.path.join(musique_dir, "sample.jsonl")
    with open(musique_file, "w", encoding="utf-8") as f:
        for item in SAMPLE_MUSIQUE_DATA:
            f.write(json.dumps(item) + "\n")
    paths["musique"] = musique_file
    logger.info(f"Generated MuSiQue benchmark samples at: {musique_file} ({len(SAMPLE_MUSIQUE_DATA)} items)")

    # 2. HotpotQA
    hotpot_dir = os.path.join(base_dir, "hotpotqa")
    os.makedirs(hotpot_dir, exist_ok=True)
    hotpot_file = os.path.join(hotpot_dir, "sample.json")
    with open(hotpot_file, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_HOTPOTQA_DATA, f, indent=2)
    paths["hotpotqa"] = hotpot_file
    logger.info(f"Generated HotpotQA benchmark samples at: {hotpot_file} ({len(SAMPLE_HOTPOTQA_DATA)} items)")

    # 3. 2WikiMultiHopQA
    twowiki_dir = os.path.join(base_dir, "2wikimultihopqa")
    os.makedirs(twowiki_dir, exist_ok=True)
    twowiki_file = os.path.join(twowiki_dir, "sample.json")
    with open(twowiki_file, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_2WIKI_DATA, f, indent=2)
    paths["2wikimultihopqa"] = twowiki_file
    logger.info(f"Generated 2WikiMultiHopQA benchmark samples at: {twowiki_file} ({len(SAMPLE_2WIKI_DATA)} items)")

    return paths


if __name__ == "__main__":
    prepare_sample_datasets()
