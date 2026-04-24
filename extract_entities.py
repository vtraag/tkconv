# %% Import libraries

import itertools
import os
import spacy
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
import sqlite3

nlp = spacy.load("nl_core_news_lg")

nlp.enable_pipe("senter")

# %% Setup path

base_dir = Path(".")

# %% Setup support functions for entity extraction

# Extract entities with sentence index
@dataclass
class Entity:
    """Class for keeping track of an extracted entity."""
    doc_id: str
    text: str
    label: str
    start_char: int = 0
    sent_idx: int = -1

def extract_entities_with_sentence_idx(doc, doc_id=-1):
    """Extract entities with their text, label, starting character, and sentence index."""
    results = []
    sents = list(doc.sents)
    sent_idx = 0

    for ent in doc.ents:
        # Move to the correct sentence for this entity
        while sent_idx < len(sents) - 1 and ent.start_char >= sents[sent_idx + 1].start_char:
            sent_idx += 1

        # Check if entity is within this sentence
        actual_sent_idx = -1
        if ent.start_char >= sents[sent_idx].start_char and ent.end_char <= sents[sent_idx].end_char:
            actual_sent_idx = sent_idx

        results.append(Entity(doc_id, ent.text, ent.label_, ent.start_char, actual_sent_idx))

    return results

# %% Function to strip HTML tags from text

# Source - https://stackoverflow.com/a/925630
# Posted by Eloff, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-24, License - CC BY-SA 4.0

from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

# %% Connect to index and entities database

con = sqlite3.connect(base_dir / "tkentities.sqlite3")
con.execute("ATTACH DATABASE 'tkindex.sqlite3' AS tkindex")

# Create entities table if it doesn't exist
print("Setting up database...")
con.execute("""
CREATE TABLE IF NOT EXISTS main.entities (
    uuid TEXT,
    text TEXT,
    label TEXT,
    start_char INTEGER,
    sent_idx INTEGER)
""")
con.execute("CREATE INDEX IF NOT EXISTS idx_entities_uuid  ON entities (uuid);")
con.execute("CREATE INDEX IF NOT EXISTS idx_entities_label ON entities (label);")
con.execute("CREATE INDEX IF NOT EXISTS idx_entities_text  ON entities (text, label);")

# %% Get documents that have not yet been processed

print("Retrieving documents to process.")
res = con.execute(
    """
    SELECT tekst, uuid
    FROM tkindex.docsearch
    WHERE NOT EXISTS (SELECT 1 FROM main.entities WHERE main.entities.uuid = tkindex.docsearch.uuid)
          AND LENGTH(tekst) < 1000000; -- Limit of nlp.max_length due to memory limitations
    """)

print("Retrieved document to process.")

# %% Process all documents

# We can process the documents as tuples, with the text first and the context later, in this case the uuid
docs = nlp.pipe(res, as_tuples=True,
                n_process=max(os.process_cpu_count() - 2, 1), batch_size=1,
                disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

with tqdm(desc="Processing documents") as t:
    for doc, doc_id in docs:
        ents = extract_entities_with_sentence_idx(doc, doc_id=doc_id)
        with con: # This ensures that the transaction is committed if successful
            res = con.executemany(
                """
                    INSERT INTO main.entities
                    VALUES (:doc_id, :text, :label, :start_char, :sent_idx)
                """,
                [asdict(e) for e in ents]
            )
        t.update(1)

# %% Close connection when done
con.close()
