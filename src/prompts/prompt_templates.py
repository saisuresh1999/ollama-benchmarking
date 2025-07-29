def anomaly_prompt(lines):
    """
    Prompt for LLM anomaly detection (for highly imbalanced synthetic data).
    """
    gl_text = "\n".join([
        f"Text: {r['text']}, GL Account: {r['gl_account_name']} ({r['gl_account']}), "
        f"Amount: {r['amount']}, Flag: {'Credit' if r['cd_flag']=='C' else 'Debit'}, "
        f"User: {r['user']}, Tax Rate: {r['tax_rate']:.2f}, "
        f"Promptly: {r['promptly']}, Weekend: {r['weekend']}, "
        f"NWH (Non-working hour): {r['nwh']}"
        for r in lines if "gl_account" in r
    ])
    legend = (
        "Field explanations:\n"
        "- Promptly: 1 = entry posted promptly (within 0–9 days), 0 = delay or non-prompt.\n"
        "- Weekend: 1 = posted on weekend, 0 = not weekend.\n"
        "- NWH (Non-working hour): 1 = posted outside 8AM–5PM, 0 = within business hours.\n"
        "- User: name of posting accountant. Each user usually posts specific types of transactions.\n"
        "- GL Account: The account should match the business case and user responsibility.\n"
        "- Tax Rate: Typical values are 0.00, 0.07, or 0.19.\n"
        "- Flag: D = Debit, C = Credit (should match the account and transaction type).\n"
    )
    business_context = (
        "Background:\n"
        "This data simulates a garden tools trading company with realistic accounting patterns, "
        "including regular and plausible transactions, but a very small number (~1%) of artificial anomalies. "
        "Anomalies are intentionally rare and only occur when there is a highly implausible or conflicting combination "
        "of attributes—for example, an unexpected user, wrong account for the business case, impossible posting times, "
        "or mismatched D/C flag and GL account. Typical business logic should explain almost all entries as normal.\n"
        "\n"
        "Important instructions:\n"
        "- Do NOT flag an entry as anomalous for minor or single-field deviations alone (e.g., late posting, or non-working hour alone), unless the combination is clearly implausible.\n"
        "- Use your expert knowledge: If all attributes are plausible and match typical patterns (business case, account, user, timing, tax rate), the entry is likely normal.\n"
        "- Flag as anomalous ONLY if there is a clear conflict, contradiction, or highly unlikely combination of values that a human accountant would consider suspicious, e.g.:\n"
        "    • A user posts a type of transaction they never do\n"
        "    • A GL account is used for an implausible purpose or with wrong tax rate\n"
        "    • Multiple unusual fields together (wrong user, time, and amount)\n"
        "- If in doubt, err on the side of NOT calling something anomalous—false positives are very costly for auditors.\n"
    )
    prompt = (
        "You are a highly experienced financial auditor performing journal entry anomaly detection "
        "for a simulated trading company. Carefully analyze the following multi-line journal entry.\n\n"
        f"{legend}\n"
        f"{business_context}\n"
        "Respond in **valid JSON only** and **terminate with <|endofanalysis|>**.\n"
        "Your output format (example):\n"
        "{\n"
        '  "explanation": "<one or two lines of reasoning>",\n'
        '  "anomaly": <integer 0 or 1>\n'
        "}\n"
        "Here is the entry:\n\n"
        f"{gl_text}\n"
    )
    return prompt
