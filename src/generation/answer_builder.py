from src.core.models import GeneratedAnswer, VerificationReport

def build_final_response(generated: GeneratedAnswer, report: VerificationReport) -> str:
    if report.overall_status != "PASS":
        return "Not enough verified evidence in the PDFs to answer reliably."
    return generated.short_answer
