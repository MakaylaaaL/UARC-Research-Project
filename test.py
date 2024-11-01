import numpy as np
from flair.embeddings import TransformerDocumentEmbeddings 
from flair.data import Sentence

# Simulation extracted reports (Sample text data)
sample_reports = [
    "the pilot departed on the agricultural application flight with full fuel tanks and 105 gallons of spray",
    "mixture onboard which he stated was a lighter load than the previous day he described the airplane",
    "climb as shallow and stated that as the airplane neared the end of the runway it began to descend",
    "the left main landing gear impacted a power line and the airplane subsequently descended into terrain",
    "resulting in substantial damage to the wings and fuselage",
    "postaccident examination of the engine revealed no preimpact mechanical malfunction or failure that",
    "would have precluded normal operation",
    "a breathalyzer test performed about 45 minutes after the accident revealed that the pilot's blood alcohol",
    "content was at the time of the accident the pilot's blood alcohol content was likely to",
    "which would have been impairing"
]

# Initialize BERT embeddings
document_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')

# Generate document embedding function
def embed_text(text):
    sentence = Sentence(text)
    document_embeddings.embed(sentence)
    return sentence.embedding.detach().cpu().numpy()  # Ensure it's moved to CPU and converted to NumPy array

# Generate embeddings for each sample report
report_vectors = []
for report in sample_reports:
    vector = embed_text(report)
    report_vectors.append(vector)
    print(f"Processed report: {report}")

# Print the resulting vectors (for demonstration)
print("\nDocument Embeddings (Vectors):")
for i, vector in enumerate(report_vectors):
    print(f"Report {i + 1} vector:\n", vector)
