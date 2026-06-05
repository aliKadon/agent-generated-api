"""
tools/router.py — generates the run_tools() function source that gets
embedded in every produced agent file.
"""

from tools.registery import ALL_TOOLS


def build_tool_router_code(selected_tools: list[str]) -> str:
    """
    Return a standalone run_tools(user_input) -> str function as a source
    string, containing only the branches for the requested tools.
    """
    if not selected_tools:
        return 'def run_tools(user_input: str) -> str:\n    return ""\n'

    blocks: list[str] = []

    for tool in selected_tools:
        info = ALL_TOOLS.get(tool)
        if not info:
            continue
        keywords = info.get("trigger_keywords", [])
        kw_check = " or ".join([f'"{kw}" in user_lower' for kw in keywords])

        if tool == "web_search":
            blocks.append(f'''\
    # web_search — DuckDuckGo, free, no key
    if {kw_check}:
        _r = tool_web_search(user_input)
        if _r:
            context_parts.append("[Web Search]\\n" + _r)
''')

        elif tool == "calculator":
            blocks.append('''\
    # calculator
    _calc_m = _re.search(r\'[\\d][\\d\\s+\\-*/^().]+[\\d]\', user_input)
    if _calc_m or any(w in user_lower for w in ["calculate", "compute", "math", "how much"]):
        _expr = _calc_m.group() if _calc_m else user_input
        context_parts.append("[Calculator]\\n" + tool_calculator(_expr))
''')

        elif tool == "datetime":
            blocks.append(f'''\
    # datetime
    if {kw_check}:
        context_parts.append("[Date & Time]\\n" + tool_datetime())
''')

        elif tool == "file_reader":
            blocks.append('''\
    # file_reader
    _path_m = _re.search(r\'[\\w./\\\\-]+\\.\\w+\', user_input)
    if _path_m and any(w in user_lower for w in ["read", "open", "load", "file"]):
        context_parts.append("[File]\\n" + tool_file_reader(_path_m.group()))
''')

        elif tool == "pdf_generator":
            blocks.append('''\
    # pdf_generator — call tool_pdf_generator(text, filename) manually after response
''')

        elif tool == "memory":
            blocks.append('''\
    # memory — ADK InMemoryMemoryService, semantic, free, no API key
    _name_m = _re.search(
        r\'(?:my name(?:\\s+is)?|call me|i am|im|remember me as)\\s+([\\w]+)\',
        user_input, _re.IGNORECASE
    )
    if _name_m:
        tool_memory_store(f"User\'s name is {_name_m.group(1)}")
        context_parts.append(f"[Memory] Stored: user name = {_name_m.group(1)}")
    _fact_m = _re.search(
        r\'remember(?:\\s+that)?\\s+(.+?)\\s+(?:is|=)\\s+(.+)\',
        user_input, _re.IGNORECASE
    )
    if _fact_m:
        tool_memory_store(f"{_fact_m.group(1).strip()} is {_fact_m.group(2).strip()}")
        context_parts.append(f"[Memory] Stored: {_fact_m.group(1).strip()} = {_fact_m.group(2).strip()}")
    if _re.search(r\'what[\\W]*(is|s) my|do you know my|recall|what did i (tell|say|mention)\', user_lower):
        _mem_r = tool_memory_search(user_input)
        if _mem_r and "Nothing" not in _mem_r:
            context_parts.append("[Memory - What I remember]\\n" + _mem_r)
    elif any(w in user_lower for w in ["my", "i ", "i\'m", "me"]):
        _mem_r = tool_memory_search(user_input)
        if _mem_r and "Nothing" not in _mem_r:
            context_parts.append("[Memory - Relevant context]\\n" + _mem_r)
''')

        elif tool == "translator":
            blocks.append(f'''\
    # translator — MyMemory free API, no key, 5000 chars/day
    if {kw_check}:
        _lang_m = _re.search(r\'in (\\w+)\', user_input, _re.IGNORECASE)
        _lang   = _lang_m.group(1)[:2].lower() if _lang_m else "en"
        _txt    = _re.sub(r\'translate\\s*\', "", user_input, flags=_re.IGNORECASE)
        _txt    = _re.sub(r\'in \\w+$\', "", _txt).strip()
        context_parts.append("[Translation]\\n" + tool_translator(_txt, _lang))
''')

        elif tool == "weather":
            blocks.append(f'''\
    # weather — wttr.in free API, no key
    if {kw_check}:
        _city_m = _re.search(r\'(?:in|for)\\s+([A-Za-z ]+)\', user_input)
        _city   = _city_m.group(1).strip() if _city_m else user_input
        context_parts.append("[Weather]\\n" + tool_weather(_city))
''')

        elif tool == "image_saver":
            # image_saver has no trigger keywords — it is called directly
            # inside the text_to_image inference body, NOT via run_tools().
            blocks.append('''\
    # image_saver — called directly after text_to_image(), not via keyword trigger
''')

        elif tool == "image_upscaler":
            blocks.append(f'''\
    # image_upscaler — Replicate free tier, needs REPLICATE_API_TOKEN in .env
    if {kw_check}:
        _img_m = _re.search(r\'[\\w./\\\\-]+\\.png\', user_input)
        if _img_m:
            context_parts.append("[Upscaler]\\n" + tool_image_upscaler(_img_m.group()))
        else:
            context_parts.append("[Upscaler] Please provide the image path to upscale.")
''')

    router_body = "\n".join(blocks) if blocks else "    pass\n"

    return (
        "def run_tools(user_input: str) -> str:\n"
        '    """Run relevant tools and return combined context string."""\n'
        "    import re as _re\n"
        "    context_parts = []\n"
        "    user_lower = user_input.lower()\n\n"
        + router_body
        + '\n    return "\\n".join(context_parts)\n'
    )