# Recorded language-model replies

`*_prompt.txt` is the prompt produced by `stochlift.llm.build_prompt` for the two examples.
`*_reply.txt` is the unedited reply of a language model to that prompt, recorded once on
2026-09-30 (Claude Haiku, one attempt per prompt, no retries). The tests replay these replies;
they do not call any API.

Two replies are a smoke test, not an accuracy measurement. The farmer prompt names a textbook
problem, so that reply may rely on the model's memory of the textbook.
