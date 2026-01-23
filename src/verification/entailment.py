from transformers import pipeline

class EntailmentScorer:
    def __init__(self, model_name: str = "roberta-large-mnli"):
        # returns: LABEL_0/1/2 depending on model, but pipeline gives score per label
        self.pipe = pipeline("text-classification", model=model_name, top_k=None)

    def score(self, claim: str, evidence: str) -> float:
        """
        Returns support score 0..1 based on entailment probability.
        """
        # NLI expects: premise (evidence) and hypothesis (claim)
        out = self.pipe({"text": evidence, "text_pair": claim})

        # out example: [{'label': 'ENTAILMENT', 'score': 0.91}, ...]
        entail_score = 0.0
        for item in out:
            label = item["label"].upper()
            if "ENTAIL" in label:
                entail_score = float(item["score"])
                break
        return entail_score
