"""Generate 100 difficult legal queries based on the CITIOM v Gulfstream case."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# These queries are designed to test different aspects of legal reasoning:
# - Factual extraction
# - Multi-document synthesis
# - Timeline construction
# - Contradiction detection
# - Legal analysis
# - Evidence assessment

LEGAL_QUERIES = [
    # ========== Category 1: Factual Extraction (20 queries) ==========
    "What was Gulfstream's initial cost estimate for the 192-month inspection?",
    "When did CITIOM first deliver the aircraft to Gulfstream for inspection?",
    "What is the serial number of the aircraft in dispute?",
    "Who is Bassim Haider and what role did he play in this dispute?",
    "What was the originally promised return-to-service date?",
    "How many days did the aircraft actually remain out of service?",
    "What was the final cost billed by Gulfstream for the inspection?",
    "Who is Danny Farnham and what was his expert opinion?",
    "What specific workscope items were included in the 192-month inspection?",
    "What was the estimated downtime in the original proposal Q-11876?",
    "What historical average downtime did Gulfstream's St. Louis facility have for similar inspections?",
    "When was Customer Support Proposal Q-11876 issued?",
    "What model aircraft is at issue in this case?",
    "Who is Charles Philipp and what was his role at Gulfstream?",
    "What was the total variance between estimated and actual costs?",
    "What percentage did actual costs exceed the original estimate?",
    "When did Gulfstream implement new forecasting analytics?",
    "What was Joe Rivera's position at Gulfstream?",
    "How many man-hours were originally estimated for the inspection?",
    "What date did CITIOM file its Statement of Claim?",

    # ========== Category 2: Multi-Document Synthesis (20 queries) ==========
    "What evidence across multiple documents supports that Gulfstream knew its estimates were unreliable?",
    "How do the expert reports from both sides differ in their assessment of damages?",
    "Synthesize all communications regarding scope changes during the inspection.",
    "What documents corroborate CITIOM's claim of misrepresentation?",
    "Compare the testimony in witness statements with documentary evidence about downtime estimates.",
    "What pattern emerges from examining all invoices and change orders?",
    "How do the cross-examination outlines suggest contradictions in Gulfstream's position?",
    "What do the direct examination outlines reveal about CITIOM's key arguments?",
    "Synthesize all references to the 'analytics' Gulfstream developed in 2023.",
    "What evidence links the painting costs to the overall dispute?",
    "Compare the Pre-Hearing Briefs from both parties on the estimate issue.",
    "What do all witness statements say about communication between parties?",
    "Synthesize the timeline of estimate revisions across all documents.",
    "What documents discuss the quality of work performed?",
    "Compare expert assessments of Gulfstream's estimating methodology.",
    "What evidence from multiple sources addresses industry standards?",
    "Synthesize all references to change orders and their justifications.",
    "What patterns emerge in how Gulfstream communicated estimate changes?",
    "Compare documentation of promised vs actual delivery dates across sources.",
    "What do multiple documents reveal about Gulfstream's internal processes?",

    # ========== Category 3: Timeline Construction (15 queries) ==========
    "Construct a complete timeline of the 192-month inspection from proposal to final invoice.",
    "What is the chronology of all estimate revisions?",
    "Map the sequence of events from aircraft delivery to arbitration filing.",
    "When were each of the major scope changes identified and communicated?",
    "What is the timeline of expert report submissions?",
    "Chronologically trace all communications about delays.",
    "When did Gulfstream first acknowledge the estimate would be exceeded?",
    "What dates are critical to establishing the misrepresentation claim?",
    "Build a timeline of all contract modifications and amendments.",
    "When were key personnel changes made during the inspection?",
    "What is the sequence of discovery requests and responses?",
    "Map the timeline of all work stoppages or delays.",
    "When did each party become aware of the cost overrun issue?",
    "Trace the chronological development of the damages calculation.",
    "What is the timeline of all formal notices between parties?",

    # ========== Category 4: Contradiction Detection (15 queries) ==========
    "Are there contradictions between Gulfstream's proposal and actual work performed?",
    "Do witness statements contradict documentary evidence?",
    "What inconsistencies exist between Gulfstream's position and its internal documents?",
    "Are there contradictions in how different Gulfstream employees described the estimate?",
    "Do the expert reports identify contradictions in Gulfstream's methodology?",
    "What conflicts exist between promised and actual work completion dates?",
    "Are there inconsistencies in how Gulfstream justified cost increases?",
    "Do cross-examination materials reveal contradictions in testimony?",
    "What discrepancies exist between verbal representations and written proposals?",
    "Are there contradictions between the original scope and final workscope?",
    "What inconsistencies exist in Gulfstream's explanation of delays?",
    "Do different documents give conflicting information about the same events?",
    "Are there contradictions in how damages were calculated by different experts?",
    "What conflicts exist between contract terms and actual performance?",
    "Do internal Gulfstream documents contradict public communications?",

    # ========== Category 5: Legal Analysis (15 queries) ==========
    "What legal elements must CITIOM prove to establish misrepresentation?",
    "What defenses does Gulfstream assert and what evidence supports them?",
    "How does the evidence support or undermine CITIOM's damages claim?",
    "What is the legal significance of the proposal being an 'estimate'?",
    "How might Gulfstream's historical data affect the misrepresentation claim?",
    "What duty of care, if any, did Gulfstream owe regarding its estimates?",
    "How do industry standards factor into assessing reasonableness?",
    "What is the evidentiary strength of the expert opinions?",
    "How might the timing of Gulfstream's analytics impact liability?",
    "What legal remedies is CITIOM seeking and are they supported?",
    "How does the contract language affect interpretation of the estimate?",
    "What procedural issues exist in the arbitration?",
    "How credible are the key witnesses based on documentary evidence?",
    "What burden of proof applies and how is it being met?",
    "How might the Tribunal weigh conflicting expert opinions?",

    # ========== Category 6: Evidence Assessment (15 queries) ==========
    "What is the strongest evidence supporting CITIOM's position?",
    "What evidence is most damaging to Gulfstream's defense?",
    "Which citations are most critical to the damages calculation?",
    "What gaps exist in CITIOM's evidentiary presentation?",
    "Which documents would be most compelling to a tribunal?",
    "What hearsay issues might affect the admissibility of evidence?",
    "Which expert's methodology is better supported by documents?",
    "What authentication issues might affect key exhibits?",
    "Which witness testimony is most strongly corroborated?",
    "What circumstantial evidence supports the fraud allegation?",
    "Which documents best establish Gulfstream's knowledge?",
    "What evidence chain establishes reliance by CITIOM?",
    "Which citations demonstrate breach of duty?",
    "What evidence is most vulnerable to challenge?",
    "Which documents are most likely to be dispositive?",
]

def save_queries():
    """Save queries to JSON file."""
    output = {
        "total": len(LEGAL_QUERIES),
        "categories": {
            "factual_extraction": LEGAL_QUERIES[0:20],
            "multi_document_synthesis": LEGAL_QUERIES[20:40],
            "timeline_construction": LEGAL_QUERIES[40:55],
            "contradiction_detection": LEGAL_QUERIES[55:70],
            "legal_analysis": LEGAL_QUERIES[70:85],
            "evidence_assessment": LEGAL_QUERIES[85:100],
        },
        "all_queries": LEGAL_QUERIES,
    }

    with open("legal_queries.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(LEGAL_QUERIES)} legal queries")
    print(f"Categories:")
    for cat, queries in output["categories"].items():
        print(f"  {cat}: {len(queries)}")

    return LEGAL_QUERIES


if __name__ == "__main__":
    queries = save_queries()
