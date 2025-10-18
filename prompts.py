# Prompts for VLM detector

# Direct prompting
prompt_direct = (
    "Is this image edited or modified by AI? Start your answer with 'Yes' or 'No'."
)

# CoT prompting
prompt_cot = "Think step-by-step to decide whether this image is edited or modified by AI. Then conclude your answer with 'The answer is' followed by your final answer."

# Judge model prompt
prompt_judge = """Read the following description and determine the expressed attitude toward whether the image is edited or modified by AI:
---
<<RESPONSE>>
---

The attitude should be summarized as one of:
- 'Yes': the description affirms it is edited or modified by AI
- 'No': the description denies it is edited or modified by AI
- 'Not sure': the description expresses uncertainty

Examples:
Description: "yes, the image is edited or modified by AI."
Answer: Yes

Description: "the image is not edited by ai"
Answer: No

Description: "can not decide without further information"
Answer: Not sure

Respond only with: 'Yes', 'No', or 'Not sure'."""
