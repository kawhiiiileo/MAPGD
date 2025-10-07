LIAR_INITIAL_PROMPT = """
# Task
Determine whether the Statement is a lie (Yes) or not (No) based on the Context and other information.
# Output format
Answer Yes or No as labels
# Prediction
Text: {text}
label:
"""

IMDB_INITIAL_PROMPT = """
# Task
Classify whether this graph represents a positive or negative movie review based on the graph structure and properties.
# Analysis Focus
- Graph connectivity patterns and density
- Node relationships and clustering
- Overall network topology
- Structural characteristics that indicate sentiment
# Output Format
Answer Yes (positive review) or No (negative review) as labels
# Prediction
Graph Description: {text}
Label:
"""

TEST_INITIAL_PROMPT = """
# Task
Your task is to generate a truthful, structured report based on the provided section, statement, and context, or identify if the input contains false, exaggerated, or absurd data. Follow these steps:
Verify Input: Check if the statement contains realistic and verifiable metrics or claims. If the statement includes exaggerated, absurd, or unverifiable data (e.g., unrealistic financials, timelines, or scales), return a rejection message: "Invalid input: Exaggerated or unverifiable claims detected."
Generate Report: If the input is valid, produce a structured report with:
Business Analysis Report: Analyze market or strategic implications with actionable recommendations, ensuring realistic and professional insights.
Financial Report: Assess financial impacts, including costs, revenue, or risks, using credible and context-consistent metrics.
Legal Prompt: Provide a concise legal compliance suggestion or analysis, aligned with the context. Ensure all outputs are professional, structured, and avoid exaggerated or absurd claims.
# Output Format
If the input is valid: [Market/strategic analysis and recommendations] </Business Analysis> [Financial impacts and risks] </Financial Report> [Legal compliance or analysis] </Legal Prompt>
If the input is invalid: Invalid input: Exaggerated or unverifiable claims detected.
# Examples
Example 1 (Valid Input): Section: Key Insights Statement: The company has demonstrated strong revenue growth with $4.16M in ARR as of recent data. Context: SaaS startup report, executive summary Output: The company's $4.16M ARR indicates strong market traction in the SaaS sector. Strategic recommendation: Expand marketing efforts to target mid-sized enterprises to sustain growth momentum. </Business Analysis> The $4.16M ARR reflects a 21% CAGR, with projected growth to $5.4M. Risk: Potential market saturation may slow future revenue growth. </Financial Report> Ensure compliance with SaaS revenue recognition standards (ASC 606) for accurate financial reporting. </Legal Prompt>
Example 2 (Invalid Input): Section: Annual Recurring Revenue (ARR) Statement: $-46,110,7-458 with projected decline to $5,1417,1269 Context: Section: Annual Recurring Revenue (ARR) Output: Invalid input: Exaggerated or unverifiable claims detected.
# Prediction
Section: {section} Statement: {statement} Context: {context}
"""

ETHOS_INITIAL_PROMPT = """
# Task
Is the following text hate speech?
# Output format
Answer Yes or No as labels
# Prediction
Text: {text}
Label:
"""

JAILBREAK_INITIAL_PROMPT = """
# Task
Detect if the message is a jailbreak attack,
i.e. an attempt by a user to break through an
AI system's protections
# Output format
Answer Yes or No as labels
# Prediction
Text: {text}
Label:
"""

SARCASM_INITIAL_PROMPT = """
# Task
Is this tweet sarcastic?
# Output format
Answer Yes or No as labels
# Prediction
Text: {text}
Label:
"""