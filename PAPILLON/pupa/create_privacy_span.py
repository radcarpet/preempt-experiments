import openai
import re


client = openai.OpenAI(api_key="api-key")

prompt_text = {
    "name": open("./prompts/extract_privacy_span_translation_task_name.txt").read(),
    "age": open("./prompts/extract_privacy_span_translation_task_age.txt").read(),
    "money": open("./prompts/extract_privacy_span_translation_task_money.txt").read()
}

def generate_extract(og, redacted):
    msgs = [{"role": "system", "content": f"Given the original string and the redacted string, what are the contents of the [REDACTED] segments? Give your answers one line per segment.\n\nORIGINAL: {og}\n\nREDACTED: {redacted}"}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs
    )
    return list(set(response.choices[0].message.content.lower().split("\n")))


def redact_text(user_prompt, entity):
    # print("PROMPT TEXT:", prompt_text[entity])
    msgs = [{"role": "system", "content": prompt_text[entity] + user_prompt}]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=msgs
    )
    return response.choices[0].message.content

def unredact_information(original_query, redacted):
    redact_segment = redacted.split("[REDACTED]")
    unredact_segments = []
    for i in range(len(redact_segment) - 1):
        try:
            redact_info = re.search(fr"{redact_segment[i]}[\s\S]*?{redact_segment[i + 1]}", original_query).group(0)
        except AttributeError:
            return generate_extract(original_query, redacted)
        except re.error:
            return generate_extract(original_query, redacted)
        unredaction = redact_info[len(redact_segment[i]):]
        unredaction = unredaction[:-len(redact_segment[i + 1])]
        if len(unredaction.split(" ")) >= 5:
            return generate_extract(original_query, redacted)
        unredact_segments.append(unredaction.lower().strip())
    unredact_segments = list(set(unredact_segments))
    return unredact_segments

def process_user_query(query, entity):
    user_query_redacted = redact_text(query, entity)
    # print("\nREDACTED_QUERY", user_query_redacted)
    if user_query_redacted and "[REDACTED]" in user_query_redacted:
        all_redacted_spans = unredact_information(query, user_query_redacted)
        if len(all_redacted_spans):
            pii_units = "||".join(all_redacted_spans)
        else:
            pii_units = None
    else:
        pii_units = None
    return pii_units, user_query_redacted

