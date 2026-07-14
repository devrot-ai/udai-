"""Donut extractor — OCR-free visual document understanding.

Uses the DocVQA-fine-tuned Donut checkpoint and asks one natural-language
question per schema field (e.g. "What is the invoice number?") directly
against the rendered page image — no OCR step at all. Confidence is the
exponentiated mean log-probability of the generated answer tokens.

Loaded lazily; reports unavailable when torch/transformers are missing so the
hybrid pipeline can fall through.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ...config import settings
from ..ingest import IngestedDocument, render_pdf_page
from .base import ExtractedField, attach_location

logger = logging.getLogger(__name__)

_QUESTIONS: Dict[str, Dict[str, str]] = {
    "invoice": {
        "invoice_number": "What is the invoice number?",
        "invoice_date": "What is the invoice date?",
        "due_date": "What is the due date?",
        "vendor_name": "Who issued this invoice?",
        "client_name": "Who is this invoice billed to?",
        "total_amount": "What is the total amount due?",
        "tax": "What is the tax amount?",
        "payment_terms": "What are the payment terms?",
    },
    "contract": {
        "party_1": "Who is the first party of this agreement?",
        "party_2": "Who is the second party of this agreement?",
        "effective_date": "What is the effective date of this agreement?",
        "governing_law": "Which state law governs this agreement?",
        "contract_value": "What is the contract value?",
    },
    "financial_statement": {
        "company_name": "What company is this statement for?",
        "period": "What period does this statement cover?",
        "total_revenue": "What is the total revenue?",
        "net_income": "What is the net income?",
        "total_assets": "What are the total assets?",
    },
    "court_document": {
        "case_number": "What is the case number?",
        "plaintiff": "Who is the plaintiff?",
        "defendant": "Who is the defendant?",
        "filing_date": "When was this document filed?",
    },
}
_QUESTIONS["receipt"] = _QUESTIONS["invoice"]

_MODEL_CACHE: dict = {}


class DonutExtractor:
    name = "donut"

    def __init__(self) -> None:
        self._checked: Optional[bool] = None

    def available(self) -> bool:
        if not settings.enable_donut:
            return False
        if self._checked is None:
            try:
                import torch  # noqa: F401
                from transformers import VisionEncoderDecoderModel  # noqa: F401

                self._checked = True
            except Exception:
                self._checked = False
        return self._checked

    def _load(self):
        if "model" not in _MODEL_CACHE:
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            logger.info("Loading Donut checkpoint %s", settings.donut_model)
            _MODEL_CACHE["processor"] = DonutProcessor.from_pretrained(settings.donut_model)
            _MODEL_CACHE["model"] = VisionEncoderDecoderModel.from_pretrained(settings.donut_model)
            _MODEL_CACHE["model"].eval()
        return _MODEL_CACHE["processor"], _MODEL_CACHE["model"]

    def extract(self, doc: IngestedDocument, doc_type: str) -> List[ExtractedField]:
        if not self.available():
            return []
        questions = _QUESTIONS.get(doc_type)
        if not questions:
            return []
        try:
            return self._extract(doc, questions)
        except Exception as exc:  # pragma: no cover
            logger.exception("Donut extraction failed: %s", exc)
            return []

    def _first_page_image(self, doc: IngestedDocument):
        page = doc.pages[0]
        if page.image is not None:
            return page.image.convert("RGB")
        if doc.path.lower().endswith(".pdf"):
            return render_pdf_page(doc.path, 1, scale=2.0).convert("RGB")
        return None

    def _extract(self, doc: IngestedDocument, questions: Dict[str, str]) -> List[ExtractedField]:
        import torch

        processor, model = self._load()
        image = self._first_page_image(doc)
        if image is None:
            return []
        pixel_values = processor(image, return_tensors="pt").pixel_values

        results: List[ExtractedField] = []
        for field_name, question in questions.items():
            prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
            decoder_input_ids = processor.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            with torch.no_grad():
                out = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=model.decoder.config.max_position_embeddings,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            seq = processor.batch_decode(out.sequences)[0]
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(
                processor.tokenizer.pad_token, ""
            )
            m = re.search(r"<s_answer>(.*)$", seq, re.DOTALL)
            answer = re.sub(r"<.*?>", "", m.group(1)).strip() if m else ""
            if not answer or answer.lower() in ("none", "n/a", "-"):
                continue

            # confidence: exp(mean log-prob) of generated tokens
            scores = out.scores
            if scores:
                logprobs = []
                gen_tokens = out.sequences[0][decoder_input_ids.shape[1]:]
                for step, tok in zip(scores, gen_tokens):
                    lp = torch.log_softmax(step[0], dim=-1)[tok]
                    logprobs.append(lp.item())
                conf = float(torch.exp(torch.tensor(sum(logprobs) / max(len(logprobs), 1))))
            else:
                conf = 0.5

            f = ExtractedField(
                name=field_name, value=answer,
                confidence=round(min(conf, 0.99), 3), source=self.name, page=1,
            )
            results.append(attach_location(doc, f))
        return results
