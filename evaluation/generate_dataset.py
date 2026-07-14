"""Generate the evaluation dataset: 24 synthetic legal/financial PDFs
(6 invoices, 6 contracts, 6 financial statements, 6 court documents) with
field-level ground truth JSON. Optionally rasterizes a subset into scanned
PNG variants for the OCR path.

Usage:
    python evaluation/generate_dataset.py [--out evaluation/dataset] [--scanned]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

random.seed(42)

VENDORS = [
    "Northwind Consulting LLC", "Meridian Software Corp.", "Blue Harbor Analytics Inc.",
    "Cascade Legal Services LLP", "Ironwood Capital Partners", "Summit Ridge Technologies",
]
CLIENTS = [
    "Acme Manufacturing Inc.", "Atlas Logistics LLC", "Pinnacle Healthcare Group",
    "Redwood Retail Corp.", "Lakeside Energy Company", "Falcon Media Holdings",
]
PEOPLE = [
    "John Martinez", "Sarah Chen", "David Okafor", "Emily Rousseau",
    "Michael Tanaka", "Laura Whitfield", "Robert Alvarez", "Priya Natarajan",
]
STATES = ["Delaware", "New York", "California", "Texas", "Illinois", "Washington"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _date(y: int, m: int, d: int) -> str:
    return f"{MONTHS[m - 1]} {d}, {y}"


# ─────────────────────────── PDF writers ───────────────────────────

class Doc:
    def __init__(self, path: Path):
        self.c = canvas.Canvas(str(path), pagesize=LETTER)
        self.w, self.h = LETTER
        self.y = self.h - inch

    def line(self, text: str, size: int = 10, bold: bool = False,
             indent: float = 1.0, gap: float = 0.22):
        if self.y < inch:
            self.c.showPage()
            self.y = self.h - inch
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.drawString(indent * inch, self.y, text)
        self.y -= gap * inch

    def center(self, text: str, size: int = 13, bold: bool = True, gap: float = 0.3):
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.drawCentredString(self.w / 2, self.y, text)
        self.y -= gap * inch

    def gap(self, amount: float = 0.2):
        self.y -= amount * inch

    def table(self, rows, col_xs, size: int = 9):
        for row in rows:
            if self.y < inch:
                self.c.showPage()
                self.y = self.h - inch
            for x, cell in zip(col_xs, row):
                self.c.setFont("Helvetica", size)
                self.c.drawString(x * inch, self.y, str(cell))
            self.y -= 0.22 * inch

    def save(self):
        self.c.save()


def make_invoice(path: Path, i: int) -> dict:
    vendor = VENDORS[i % len(VENDORS)]
    client = CLIENTS[i % len(CLIENTS)]
    number = f"INV-2024-{100 + i * 7}"
    m = (i % 10) + 1
    inv_date = _date(2024, m, 3 + i)
    due_date = _date(2024, m + 1, 3 + i)
    items = [
        ("Consulting services", 10 + i, 150.0 + 10 * i),
        ("Implementation support", 4 + i, 200.0),
        ("License subscription", 1, 900.0 + 50 * i),
    ]
    subtotal = sum(q * p for _, q, p in items)
    tax = round(subtotal * 0.08, 2)
    total = round(subtotal + tax, 2)
    terms = ["Net 30", "Net 15", "Net 45", "Due on receipt"][i % 4]

    d = Doc(path)
    d.line(vendor, size=14, bold=True)
    d.line("1200 Commerce Street, Suite 400")
    d.gap()
    d.center("INVOICE", size=16)
    d.gap(0.1)
    d.line(f"Invoice Number: {number}", bold=True)
    d.line(f"Invoice Date: {inv_date}")
    d.line(f"Due Date: {due_date}")
    d.line(f"Payment Terms: {terms}")
    d.gap()
    d.line("Bill To:", bold=True)
    d.line(client)
    d.line("400 Industrial Parkway")
    d.gap()
    d.table(
        [("Description", "Qty", "Unit Price", "Amount")] +
        [(desc, q, _money(p), _money(q * p)) for desc, q, p in items],
        col_xs=[1.0, 4.2, 5.0, 6.2],
    )
    d.gap(0.15)
    d.line(f"Subtotal: {_money(subtotal)}", indent=4.5)
    d.line(f"Tax (8%): {_money(tax)}", indent=4.5)
    d.line(f"Total Due: {_money(total)}", bold=True, indent=4.5)
    d.gap()
    d.line("Please remit payment to the account listed above. Thank you for your business.")
    d.save()

    return {
        "doc_type": "invoice",
        "fields": {
            "invoice_number": number,
            "invoice_date": inv_date,
            "due_date": due_date,
            "client_name": client,
            "subtotal": _money(subtotal),
            "tax": _money(tax),
            "total_amount": _money(total),
            "payment_terms": terms,
        },
    }


def make_contract(path: Path, i: int) -> dict:
    party1 = VENDORS[(i + 1) % len(VENDORS)]
    party2 = CLIENTS[(i + 2) % len(CLIENTS)]
    eff = _date(2024, (i % 12) + 1, 1)
    term_years = (i % 3) + 1
    value = 48000.0 + i * 12500
    law = STATES[i % len(STATES)]
    pay = ["Net 45", "Net 30", "Net 60"][i % 3]

    d = Doc(path)
    d.center("PROFESSIONAL SERVICES AGREEMENT", size=14)
    d.gap(0.1)
    d.line(f"This Agreement is entered into as of {eff}, by and between", gap=0.25)
    d.line(f'{party1} ("Provider") and {party2} ("Client").', gap=0.3)
    d.line("RECITALS", bold=True)
    d.line("WHEREAS, Provider is engaged in the business of providing professional services; and")
    d.line("WHEREAS, Client desires to engage Provider to perform such services;")
    d.line("NOW, THEREFORE, the parties agree as follows:", gap=0.3)
    d.line("1. SERVICES", bold=True)
    d.line("Provider shall perform the services described in Exhibit A attached hereto.")
    d.gap(0.1)
    d.line("2. COMPENSATION", bold=True)
    d.line(f"Client shall pay Provider a total fee of {_money(value)} for the services,")
    d.line(f"payable in monthly installments. Payment Terms: {pay}.")
    d.gap(0.1)
    d.line("3. TERM AND TERMINATION", bold=True)
    d.line(f"This Agreement shall commence on the Effective Date and continue for an initial")
    d.line(f"term of {term_years} year(s). Either party may terminate upon sixty (60) days written notice.")
    d.gap(0.1)
    d.line("4. CONFIDENTIALITY", bold=True)
    d.line("Each party shall maintain the confidentiality of the other party's proprietary information")
    d.line("and shall not disclose such information to any third party without prior written consent.")
    d.gap(0.1)
    d.line("5. GOVERNING LAW", bold=True)
    d.line(f"This Agreement shall be governed by and construed in accordance with the laws of")
    d.line(f"the State of {law}, without regard to its conflict of laws principles.")
    d.gap(0.1)
    d.line("6. INDEMNIFICATION", bold=True)
    d.line("Provider shall indemnify and hold harmless Client from any claims arising out of")
    d.line("Provider's breach of its obligations under this Agreement.")
    d.gap(0.2)
    d.line("IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.")
    d.gap(0.2)
    d.line(f"{party1}                                {party2}", bold=True)
    d.save()

    return {
        "doc_type": "contract",
        "fields": {
            "party_1": party1,
            "party_2": party2,
            "effective_date": eff,
            "governing_law": law,
            "contract_value": _money(value),
            "payment_terms": pay,
        },
    }


def make_financial(path: Path, i: int) -> dict:
    company = (VENDORS + CLIENTS)[(i * 2) % 12]
    period = _date(2024, 12, 31) if i % 2 == 0 else _date(2024, 6, 30)
    revenue = 4_500_000 + i * 750_000
    expenses = round(revenue * 0.62)
    net_income = round(revenue * 0.18)
    assets = revenue * 3 - i * 100_000
    liabilities = round(assets * 0.42)
    equity = assets - liabilities

    d = Doc(path)
    d.center(company, size=14)
    d.center("Consolidated Income Statement and Balance Sheet", size=11, bold=False)
    d.center(f"For the Year Ended {period}", size=10, bold=False)
    d.gap()
    d.line("INCOME STATEMENT", bold=True)
    d.table([
        ("Total Revenue", _money(revenue)),
        ("Operating Expenses", _money(expenses)),
        ("Net Income", _money(net_income)),
    ], col_xs=[1.0, 5.0])
    d.gap()
    d.line("BALANCE SHEET", bold=True)
    d.table([
        ("Total Assets", _money(assets)),
        ("Total Liabilities", _money(liabilities)),
        ("Shareholders' Equity", _money(equity)),
    ], col_xs=[1.0, 5.0])
    d.gap()
    d.line("Notes: Figures are presented in accordance with GAAP. The fiscal year")
    d.line("reflects continuing operations, excluding discontinued segments.")
    d.save()

    return {
        "doc_type": "financial_statement",
        "fields": {
            "company_name": company,
            "period": period,
            "total_revenue": _money(revenue),
            "operating_expenses": _money(expenses),
            "net_income": _money(net_income),
            "total_assets": _money(assets),
            "total_liabilities": _money(liabilities),
        },
    }


def make_court(path: Path, i: int) -> dict:
    plaintiff = PEOPLE[i % len(PEOPLE)]
    defendant = CLIENTS[(i + 3) % len(CLIENTS)].upper().rstrip(".")
    case_no = f"3:24-cv-{1200 + i * 11:05d}"
    filed = _date(2024, (i % 12) + 1, 12)
    judge = PEOPLE[(i + 4) % len(PEOPLE)]
    court = ["UNITED STATES DISTRICT COURT NORTHERN DISTRICT OF CALIFORNIA",
             "UNITED STATES DISTRICT COURT SOUTHERN DISTRICT OF NEW YORK",
             "SUPERIOR COURT OF THE STATE OF WASHINGTON"][i % 3]

    d = Doc(path)
    d.center(court.split(" DISTRICT OF ")[0] if "DISTRICT OF" in court else court, size=12)
    if "DISTRICT OF" in court:
        d.center("DISTRICT OF " + court.split(" DISTRICT OF ")[1], size=12)
    d.gap()
    d.line(plaintiff.upper() + ",", bold=True)
    d.line("Plaintiff,", indent=1.5)
    d.line("v.", indent=2.0)
    d.line(defendant + ",", bold=True)
    d.line("Defendant.", indent=1.5)
    d.gap(0.1)
    d.line(f"Case No. {case_no}", bold=True, indent=4.5)
    d.line(f"Filed: {filed}", indent=4.5)
    d.gap()
    d.center("COMPLAINT FOR BREACH OF CONTRACT", size=12)
    d.gap(0.1)
    d.line(f"Plaintiff {plaintiff}, by and through undersigned counsel, alleges as follows:")
    d.line("1. This is a civil action seeking damages arising from Defendant's breach of a")
    d.line("written services agreement between the parties.")
    d.line("2. This Court has jurisdiction over this civil action pursuant to 28 U.S.C. § 1332.")
    d.line("3. Defendant failed to remit payment due under the agreement despite repeated demand.")
    d.line("WHEREFORE, Plaintiff respectfully requests judgment against Defendant for damages,")
    d.line("costs, and such other relief as the Court deems just and proper.")
    d.gap()
    d.line(f"Before the Honorable {judge}", bold=True)
    d.save()

    return {
        "doc_type": "court_document",
        "fields": {
            "case_number": case_no,
            "plaintiff": plaintiff,
            "filing_date": filed,
            "judge": judge,
        },
    }


GENERATORS = [
    ("invoice", make_invoice),
    ("contract", make_contract),
    ("financial", make_financial),
    ("court", make_court),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evaluation/dataset")
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument("--scanned", action="store_true",
                        help="also rasterize the first doc of each type to PNG")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)

    manifest = []
    for prefix, generator in GENERATORS:
        for i in range(args.per_type):
            doc_id = f"{prefix}_{i:02d}"
            pdf_path = out / f"{doc_id}.pdf"
            truth = generator(pdf_path, i)
            (out / "ground_truth" / f"{doc_id}.json").write_text(
                json.dumps(truth, indent=2)
            )
            manifest.append({"id": doc_id, "file": pdf_path.name, **truth})

    if args.scanned:
        import pypdfium2 as pdfium

        for prefix, _ in GENERATORS:
            doc_id = f"{prefix}_00"
            pdf = pdfium.PdfDocument(str(out / f"{doc_id}.pdf"))
            image = pdf[0].render(scale=200 / 72).to_pil()
            image.save(out / f"{doc_id}_scanned.png")
            pdf.close()
            truth = json.loads((out / "ground_truth" / f"{doc_id}.json").read_text())
            (out / "ground_truth" / f"{doc_id}_scanned.json").write_text(
                json.dumps(truth, indent=2)
            )
            manifest.append({"id": f"{doc_id}_scanned",
                             "file": f"{doc_id}_scanned.png", **truth})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(manifest)} documents in {out}")


if __name__ == "__main__":
    main()
