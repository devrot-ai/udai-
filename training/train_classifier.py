"""Train the supervised document-type classifier.

Builds a randomized synthetic corpus (~900 labeled samples, 6 classes
including an "other" negative class), trains TF-IDF + Logistic Regression,
evaluates on a stratified held-out split, and writes:

    models/doc_classifier.joblib          model artifact
    models/doc_classifier.metrics.json    held-out metrics
    evaluation/results/classifier_report.md

The corpus generator randomizes field labels, entities, amounts, dates,
section order and injects OCR-style character noise so the model learns the
vocabulary/structure signal of each document class rather than a template.

Usage:
    python training/train_classifier.py [--samples-per-class 150] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────── randomized text templates ───────────────────────

FIRST = ["James", "Maria", "Wei", "Aisha", "Carlos", "Elena", "Noah", "Priya",
         "Tomas", "Grace", "Omar", "Hana", "Liam", "Sofia", "Ravi", "Ingrid"]
LAST = ["Anderson", "Delgado", "Kim", "Osei", "Petrov", "Nakamura", "Silva",
        "Haddad", "Berg", "Kowalski", "Mbeki", "Larsen", "Quinn", "Sato"]
COMPANY_A = ["Northwind", "Meridian", "Blue Harbor", "Cascade", "Ironwood",
             "Summit Ridge", "Redwood", "Lakeside", "Falcon", "Atlas",
             "Pinnacle", "Sterling", "Beacon", "Crestline", "Vanguard"]
COMPANY_B = ["Consulting", "Software", "Analytics", "Logistics", "Capital",
             "Technologies", "Manufacturing", "Healthcare", "Energy", "Media",
             "Retail", "Financial", "Legal Services", "Industries"]
SUFFIX = ["LLC", "Inc.", "Corp.", "Ltd.", "LLP", "Company", "Group", "Holdings"]
STATES = ["Delaware", "New York", "California", "Texas", "Illinois",
          "Washington", "Nevada", "Georgia", "Massachusetts"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

rng = random.Random()


def company() -> str:
    return f"{rng.choice(COMPANY_A)} {rng.choice(COMPANY_B)} {rng.choice(SUFFIX)}"


def person() -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def a_date() -> str:
    styles = [
        lambda: f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {rng.randint(2019, 2026)}",
        lambda: f"{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(2019, 2026)}",
        lambda: f"{rng.randint(2019, 2026)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
    ]
    return rng.choice(styles)()


def money(lo: float = 100, hi: float = 5_000_000) -> str:
    return f"${rng.uniform(lo, hi):,.2f}"


def _pick(*options: str) -> str:
    return rng.choice(options)


def gen_invoice() -> str:
    lines = [
        company(),
        _pick("INVOICE", "TAX INVOICE", "Invoice", "COMMERCIAL INVOICE"),
        f"{_pick('Invoice Number', 'Invoice No.', 'Invoice #', 'Bill Number')}: "
        f"{_pick('INV', 'BIL', 'IN')}-{rng.randint(2019, 2026)}-{rng.randint(100, 99999)}",
        f"{_pick('Invoice Date', 'Date of Issue', 'Date')}: {a_date()}",
        f"{_pick('Due Date', 'Payment Due')}: {a_date()}",
        f"{_pick('Bill To', 'Billed To', 'Customer', 'Sold To')}: {company()}",
        f"{_pick('Payment Terms', 'Terms')}: {_pick('Net 30', 'Net 15', 'Net 45', 'Net 60', 'Due on receipt')}",
        f"{_pick('Description', 'Item')}  Qty  {_pick('Unit Price', 'Rate')}  Amount",
    ]
    for _ in range(rng.randint(1, 5)):
        qty = rng.randint(1, 40)
        price = rng.uniform(20, 900)
        lines.append(
            f"{_pick('Consulting services', 'Support hours', 'License subscription', 'Hardware unit', 'Training session', 'Maintenance plan')} "
            f"{qty} ${price:,.2f} ${qty * price:,.2f}"
        )
    lines += [
        f"Subtotal: {money(500, 90000)}",
        f"{_pick('Tax', 'Sales Tax', 'VAT')} ({rng.randint(5, 12)}%): {money(20, 8000)}",
        f"{_pick('Total Due', 'Amount Due', 'Total', 'Balance Due', 'Grand Total')}: {money(600, 99000)}",
        _pick("Please remit payment to the account listed above.",
              "Thank you for your business.",
              "Make checks payable to the vendor listed above.", ""),
    ]
    return "\n".join(lines)


def gen_contract() -> str:
    p1, p2 = company(), company()
    sections = [
        f"{_pick('PROFESSIONAL SERVICES AGREEMENT', 'MASTER SERVICES AGREEMENT', 'CONSULTING AGREEMENT', 'SOFTWARE LICENSE AGREEMENT', 'NON-DISCLOSURE AGREEMENT', 'LEASE AGREEMENT')}",
        f"This Agreement is {_pick('entered into as of', 'made effective as of', 'dated as of')} "
        f"{a_date()}, by and between {p1} "
        f"(\"{_pick('Provider', 'Contractor', 'Licensor', 'Landlord', 'Seller')}\") and {p2} "
        f"(\"{_pick('Client', 'Customer', 'Licensee', 'Tenant', 'Buyer')}\").",
        f"WHEREAS, {_pick('Provider is engaged in the business of providing professional services', 'the parties wish to set forth the terms of their engagement', 'Licensor owns certain proprietary software')}; and",
        "NOW, THEREFORE, in consideration of the mutual covenants herein, the parties agree as follows:",
    ]
    clauses = [
        f"COMPENSATION. Client shall pay a total fee of {money(5000, 900000)}, payable {_pick('monthly', 'quarterly', 'upon completion')}. Payment terms: {_pick('Net 30', 'Net 45', 'Net 60')}.",
        f"TERM AND TERMINATION. This Agreement shall continue for an initial term of {rng.randint(1, 5)} year(s). Either party may terminate upon {_pick('thirty (30)', 'sixty (60)', 'ninety (90)')} days written notice.",
        "CONFIDENTIALITY. Each party shall maintain the confidentiality of the other party's proprietary information and shall not disclose it to any third party.",
        f"GOVERNING LAW. This Agreement shall be governed by and construed in accordance with the laws of the State of {rng.choice(STATES)}.",
        "INDEMNIFICATION. Each party shall indemnify and hold harmless the other from claims arising out of its breach of this Agreement.",
        "ASSIGNMENT. Neither party may assign its rights or obligations hereunder without prior written consent.",
        f"NOTICES. All notices shall be delivered in writing to the addresses set forth above.",
    ]
    rng.shuffle(clauses)
    picked = clauses[: rng.randint(3, len(clauses))]
    sections += [f"{i + 1}. {c}" for i, c in enumerate(picked)]
    sections.append(
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above."
    )
    return "\n".join(sections)


def gen_financial() -> str:
    rev = rng.uniform(1e6, 5e8)
    lines = [
        company(),
        _pick("Consolidated Income Statement", "Statement of Operations",
              "Consolidated Balance Sheet", "Annual Financial Statements",
              "Quarterly Financial Report"),
        f"For the {_pick('Year', 'Quarter', 'Period')} Ended {a_date()}",
        "",
        f"Total Revenue ${rev:,.0f}",
        f"Cost of Goods Sold ${rev * rng.uniform(0.3, 0.5):,.0f}",
        f"Operating Expenses ${rev * rng.uniform(0.2, 0.4):,.0f}",
        f"{_pick('Net Income', 'Net Earnings')} ${rev * rng.uniform(0.05, 0.25):,.0f}",
        "",
        f"Total Assets ${rev * rng.uniform(1.5, 4):,.0f}",
        f"Total Liabilities ${rev * rng.uniform(0.5, 2):,.0f}",
        f"Shareholders' Equity ${rev * rng.uniform(0.5, 2):,.0f}",
        f"{_pick('Retained Earnings', 'Cash and Cash Equivalents')} ${rev * rng.uniform(0.1, 1):,.0f}",
        "",
        _pick("Figures are presented in accordance with GAAP.",
              "See accompanying notes to the financial statements.",
              "Prepared on a consolidated basis for the fiscal year."),
    ]
    if rng.random() < 0.5:
        lines.insert(4, f"EBITDA ${rev * rng.uniform(0.1, 0.3):,.0f}")
    return "\n".join(lines)


def gen_court() -> str:
    plaintiff = person() if rng.random() < 0.5 else company()
    defendant = company() if rng.random() < 0.7 else person()
    lines = [
        _pick("UNITED STATES DISTRICT COURT NORTHERN DISTRICT OF CALIFORNIA",
              "UNITED STATES DISTRICT COURT SOUTHERN DISTRICT OF NEW YORK",
              "SUPERIOR COURT OF THE STATE OF WASHINGTON",
              "CIRCUIT COURT OF COOK COUNTY, ILLINOIS",
              "UNITED STATES BANKRUPTCY COURT DISTRICT OF DELAWARE"),
        f"{plaintiff.upper()},",
        "Plaintiff,",
        "v.",
        f"{defendant.upper()},",
        "Defendant.",
        f"{_pick('Case No.', 'Civil Action No.', 'Docket No.')} "
        f"{rng.randint(1, 9)}:{rng.randint(19, 26)}-cv-{rng.randint(1000, 99999):05d}",
        f"Filed: {a_date()}",
        _pick("COMPLAINT FOR BREACH OF CONTRACT", "MOTION FOR SUMMARY JUDGMENT",
              "MOTION TO DISMISS", "COMPLAINT FOR DAMAGES", "ANSWER AND COUNTERCLAIM",
              "STIPULATION AND PROPOSED ORDER"),
        f"Plaintiff, by and through undersigned counsel, {_pick('alleges', 'states', 'avers')} as follows:",
        f"1. This Court has jurisdiction over this civil action pursuant to 28 U.S.C. § {rng.choice([1331, 1332, 1367])}.",
        _pick("2. Defendant failed to remit payment due under the parties' agreement.",
              "2. Venue is proper in this district under 28 U.S.C. § 1391.",
              "2. Defendant's conduct constitutes a material breach of contract."),
        "WHEREFORE, Plaintiff respectfully requests judgment against Defendant for damages, costs, and such other relief as the Court deems just and proper.",
        f"Before the Honorable {person()}",
    ]
    return "\n".join(lines)


def gen_receipt() -> str:
    total = rng.uniform(5, 400)
    lines = [
        f"{rng.choice(COMPANY_A)} {_pick('Market', 'Cafe', 'Store', 'Pharmacy', 'Hardware')}",
        _pick("RECEIPT", "SALES RECEIPT", "*** RECEIPT ***"),
        f"{a_date()} {rng.randint(8, 21)}:{rng.randint(0, 59):02d}",
        f"Cashier: {rng.choice(FIRST)}",
    ]
    for _ in range(rng.randint(2, 6)):
        lines.append(f"{_pick('Coffee', 'Notebook', 'Batteries', 'Sandwich', 'USB Cable', 'Paper Towels')}  ${rng.uniform(1, 60):,.2f}")
    lines += [
        f"TOTAL ${total:,.2f}",
        _pick(f"CASH ${total + rng.uniform(0, 20):,.2f}", f"Card ending {rng.randint(1000, 9999)}"),
        f"Change Due ${rng.uniform(0, 15):,.2f}" if rng.random() < 0.5 else "",
        _pick("Thank you for your purchase!", "Have a nice day!", "Returns within 30 days with receipt."),
    ]
    return "\n".join(l for l in lines if l)


def gen_other() -> str:
    kind = rng.randint(0, 4)
    if kind == 0:  # memo
        return "\n".join([
            "INTERNAL MEMORANDUM",
            f"To: All Staff\nFrom: {person()}\nDate: {a_date()}",
            f"Subject: {_pick('Office relocation update', 'Quarterly all-hands meeting', 'New parking policy', 'Holiday schedule')}",
            _pick("Please note that the office will be closed next Friday for maintenance.",
                  "We are excited to announce the launch of our new internal portal.",
                  "Kindly submit your timesheets by the end of the week."),
            "Please reach out with any questions.",
        ])
    if kind == 1:  # letter
        return "\n".join([
            f"Dear {person()},",
            _pick("Thank you for your interest in our graduate program.",
                  "We appreciate your feedback regarding your recent visit.",
                  "I am writing to follow up on our conversation last week."),
            _pick("We look forward to hearing from you soon.",
                  "Please do not hesitate to contact me directly."),
            f"Sincerely,\n{person()}",
        ])
    if kind == 2:  # press release
        return "\n".join([
            "FOR IMMEDIATE RELEASE",
            f"{company()} {_pick('Announces New Product Line', 'Opens Regional Office', 'Appoints New Chief Executive')}",
            f"{rng.choice(STATES)} — {a_date()} — "
            + _pick("The company today announced a major expansion of its operations.",
                    "Leadership described the initiative as a milestone for the organization."),
            "About the company: a leading provider of innovative solutions.",
            f"Media contact: {person()}",
        ])
    if kind == 3:  # meeting minutes
        return "\n".join([
            f"Meeting Minutes — {a_date()}",
            f"Attendees: {person()}, {person()}, {person()}",
            "Agenda: project status, upcoming milestones, open questions.",
            _pick("The team reviewed progress on the current sprint.",
                  "Action items were assigned and deadlines confirmed."),
            f"Next meeting scheduled for {a_date()}.",
        ])
    return "\n".join([  # product spec / report
        f"{_pick('Product Specification', 'Technical Report', 'Project Overview')}",
        f"Author: {person()}  Version {rng.randint(1, 4)}.{rng.randint(0, 9)}",
        _pick("This document describes the system architecture and interfaces.",
              "The following sections summarize findings from the field study."),
        "Section 1: Introduction. Section 2: Requirements. Section 3: Design.",
        _pick("Performance benchmarks are provided in the appendix.",
              "Refer to the glossary for definitions of key terms."),
    ])


GENERATORS = {
    "invoice": gen_invoice,
    "contract": gen_contract,
    "financial_statement": gen_financial,
    "court_document": gen_court,
    "receipt": gen_receipt,
    "other": gen_other,
}


def add_ocr_noise(text: str, rate: float = 0.02) -> str:
    """Simulate OCR errors: character substitutions, drops, case flips."""
    subs = {"o": "0", "l": "1", "e": "c", "i": "l", "s": "5", "a": "@"}
    chars = list(text)
    for i, ch in enumerate(chars):
        if rng.random() < rate:
            action = rng.random()
            low = ch.lower()
            if action < 0.5 and low in subs:
                chars[i] = subs[low]
            elif action < 0.75:
                chars[i] = ""
            else:
                chars[i] = ch.swapcase()
    return "".join(chars)


def build_corpus(samples_per_class: int) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for label, gen in GENERATORS.items():
        for i in range(samples_per_class):
            text = gen()
            if i % 3 == 0:  # a third of samples get OCR-style noise
                text = add_ocr_noise(text)
            texts.append(text)
            labels.append(label)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-out", default="models/doc_classifier.joblib")
    args = parser.parse_args()

    rng.seed(args.seed)

    import joblib
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    texts, labels = build_corpus(args.samples_per_class)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2,
            max_features=50000, lowercase=True,
        )),
        ("clf", LogisticRegression(max_iter=2000, C=4.0)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print(f"Corpus: {len(texts)} samples ({args.samples_per_class}/class, "
          f"{len(X_test)} held out)")
    print(f"Held-out accuracy: {acc:.3f}\n")
    print(report)

    out = Path(args.model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    metrics = {
        "held_out_accuracy": round(float(acc), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classes": labels_sorted,
        "confusion_matrix": cm.tolist(),
        "vectorizer": "tfidf(1-2grams, sublinear, min_df=2)",
        "model": "LogisticRegression(C=4.0)",
        "seed": args.seed,
    }
    out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))

    # markdown report
    results_dir = ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Supervised Document Classifier — Training Report",
        "",
        f"- Corpus: {len(texts)} randomized synthetic samples, {len(labels_sorted)} classes",
        f"- Model: TF-IDF (1–2 grams, sublinear) + Logistic Regression",
        f"- Held-out accuracy (20% stratified split): **{acc:.1%}**",
        "",
        "```",
        report.rstrip(),
        "```",
        "",
        "## Confusion matrix",
        "",
        "| | " + " | ".join(labels_sorted) + " |",
        "|---|" + "---|" * len(labels_sorted),
    ]
    for lab, row in zip(labels_sorted, cm):
        lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    (results_dir / "classifier_report.md").write_text("\n".join(lines) + "\n")

    print(f"Model written to {out}")
    print(f"Report written to {results_dir / 'classifier_report.md'}")

    # out-of-distribution sanity check on the PDF eval set, if present
    manifest = ROOT / "evaluation" / "dataset" / "manifest.json"
    if manifest.exists():
        from backend.app.pipeline.ingest import ingest

        entries = json.loads(manifest.read_text())
        correct = 0
        for e in entries:
            if not e["file"].endswith(".pdf"):
                continue
            doc = ingest(str(ROOT / "evaluation" / "dataset" / e["file"]))
            pred = model.predict([doc.full_text])[0]
            correct += int(pred == e["doc_type"])
        n = sum(1 for e in entries if e["file"].endswith(".pdf"))
        print(f"PDF eval-set check (out-of-template): {correct}/{n} correct")


if __name__ == "__main__":
    main()
