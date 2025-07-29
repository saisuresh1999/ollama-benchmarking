def anomaly_prompt(lines):
    """
    Build prompt string for LLM anomaly detection with field explanations.
    """
    gl_text = "\n".join([
        f"Text: {r['text']}, GL Account: {r['gl_account_name']} ({r['gl_account']}), Amount: {r['amount']}, "
        f"Flag: {'Credit' if r['cd_flag']=='C' else 'Debit'}, User: {r['user']}, Tax Rate: {r['tax_rate']:.2f}, "
        f"Promptly: {r['promptly']}, Weekend: {r['weekend']}, NWH (Non-working hour): {r['nwh']}"
        for r in lines if "gl_account" in r
    ])
    legend = (
        "Field explanations:\n"
        "- Promptly: 1=entry posted promptly, 0=delay\n"
        "- Weekend: 1=posted on weekend, 0=not weekend\n"
        "- NWH (Non-working hour): 1=outside business hours, 0=within business hours\n"
    )
    return (
        "You are an expert accountant. Given the following journal entry (possibly suspicious), "
        "explain whether it is anomalous or not.\n"
        "Respond in **valid JSON only**, and **terminate with <|endofanalysis|>**.\n"
        "Format example:\n"
        "{\n"
        '  "explanation": "<one or two lines of reasoning>",\n'
        '  "anomaly": <integer 0 or 1>\n'
        "}\n"
        f"{legend}\n"
        "Here is the entry:\n\n"
        f"{gl_text}\n"
    )
