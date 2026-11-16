"""Teacher-forced metrics and unfiltered real Session/FIFO experiments."""
import math
import resource
import time
import torch
from torch.nn import functional as F
from .data import LaneStream
from dynamath.v003a.english_data import pairs
from .model import Session, block

PROMPTS = [
    ("greeting", "Hello!"),
    ("greeting", "Good morning. How are you today?"),
    ("greeting", "It is nice to meet you."),
    ("general", "What do you like to do in your free time?"),
    ("general", "How was your day?"),
    ("general", "Can you tell me about your family?"),
    ("short_information", "What time is it?"),
    ("short_information", "Where is the nearest bus stop?"),
    ("short_information", "How much does this ticket cost?"),
    ("short_information", "May I have a glass of water, please?"),
    ("long", "I am visiting a new city for the first time this weekend, and I only have one free afternoon before I need to catch the train home. What would you suggest I do?"),
    ("long", "My friend and I are trying to plan a quiet holiday, but she prefers the mountains and I prefer the sea. We have a small budget and only three days. How can we decide?"),
    ("multi_sentence", "I missed my bus. The next one comes in an hour. What should I do while I wait?"),
    ("multi_sentence", "I started a new job yesterday. Everyone seemed friendly. I still feel nervous about tomorrow."),
    ("multi_sentence", "The weather is cold today. I forgot my coat at home. Could I borrow a jacket?"),
    ("general", "Why do you think learning English is useful?"),
    ("general", "Would you like to have dinner with me?"),
    ("short_information", "Could you explain how to get to the library?"),
    ("general", "What should I cook for dinner tonight?"),
    ("general", "Thank you for your help. See you tomorrow!"),
]

MEMORY_CASES = [
    ("My name is Alice.", "What is my name?", "Alice", "My name is Bob."),
    ("My favorite color is blue.", "What is my favorite color?", "blue", "My favorite color is red."),
    ("I live in London.", "Where do I live?", "London", "I live in Paris."),
    ("I have a cat named Tom.", "What is my cat's name?", "Tom", "I have a cat named Sam."),
    ("I work as a teacher.", "What is my job?", "teacher", "I work as a doctor."),
]


@torch.no_grad()
def teacher_forced(model, dialogues, tokenizer, batch=8):
    model.eval()
    device = next(model.parameters()).device
    stream = LaneStream(dialogues, tokenizer, batch=batch, device=device)
    s, m = model.zero(batch)
    clocks = torch.zeros(batch, device=device, dtype=torch.long)
    ce_sum = 0.
    n = correct = omitted = false = silence = eos_total = eos_correct = processed = 0
    start = time.monotonic()
    for _ in range(int(stream.lengths.max())):
        x, y, reset, active = stream.next(cycle=False)
        s, m, clocks, out = block(model, s, m, clocks, x, reset, active)
        processed += int(active.sum())
        for p, logits in enumerate(out):
            target = y if p == 0 else torch.full_like(y, 3)
            response = active & (target != 3)
            quiet = active & (target == 3)
            pred = logits.argmax(1)
            emit = (logits.max(1).values/model.config.gain >= .1) & (pred != 3)
            ce_sum += float(F.cross_entropy(logits, target, reduction="none")[response].sum())
            n += int(response.sum())
            correct += int(((pred == target) & response).sum())
            omitted += int((response & ~emit).sum())
            silence += int(quiet.sum())
            false += int((quiet & emit).sum())
            eos_total += int(((target == 2) & active).sum())
            eos_correct += int(((target == 2) & active & (pred == 2) & emit).sum())
    seconds = time.monotonic()-start
    assert processed == stream.total_slots, "evaluation discarded slots"
    return {"mode": "teacher_forced; no predicted-token feedback",
            "response_CE": ce_sum/n, "response_perplexity": math.exp(ce_sum/n),
            "response_accuracy": correct/n, "response_omission_rate": omitted/n,
            "silence_false_emission_rate": false/silence,
            "teacher_forced_EOS_accuracy": eos_correct/eos_total,
            "response_tokens": n, "silence_checks": silence,
            "input_slots": processed, "expected_input_slots": stream.total_slots,
            "seconds": seconds, "input_slots_per_second": processed/seconds,
            "response_tokens_per_second": n/seconds,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}


@torch.no_grad()
def generate(model, tokenizer, question, response_slots=96, session=None, turn=0):
    """Do not spend response budget while the external question is in FIFO.

    Emissions during input are preserved and fed back just like any other event.
    BOT is admitted at phase 0; its next two checks count in response budget.
    Existing feedback remains queued across turns; no inference-time cleanup.
    """
    model.eval()
    session = Session(model) if session is None else session
    source = f"external:{turn}:{session.t}"
    ids = [0] + tokenizer.encode(question) + [1]
    preexisting_queue = len(session.queue)
    session.enqueue(ids, source=source)
    start = time.monotonic()
    qevents, answer, trace = [], [], []
    input_ticks = response_ticks = admitted = 0
    responding = eos = False
    max_input_ticks = 3*(preexisting_queue+len(ids)+1)
    # Start phases can be 1/2 after EOS. Allow through the next lawful boundary.
    while not responding:
        event = session.tick()
        row = session.log[-1]
        trace.append(dict(row))
        input_ticks += 1
        if row["admitted"] is not None and row["admitted"][1] == source:
            admitted += 1
            if row["admitted"][0] == 1:
                responding = True
        if event is not None:
            qevents.append({"tick": row["tick"], "token_id": event})
        if input_ticks > max_input_ticks:
            raise RuntimeError("FIFO failed to admit bounded external input")
    for _ in range(response_slots*3-1):
        event = session.tick()
        row = session.log[-1]
        trace.append(dict(row))
        response_ticks += 1
        if event is not None:
            answer.append(event)
            if event == 2:
                eos = True
                break
    seconds = time.monotonic()-start
    snapshot = session.snapshot()
    # Trace, IDs and final pending FIFO are raw evidence. Large numeric S lives
    # only in memory/checkpoint, not repeated 20+ times in JSON.
    result = {"question": question, "text": tokenizer.decode(answer), "token_ids": answer,
        "raw_tokens": [tokenizer.vocabulary[i] for i in answer],
        "question_events": qevents, "eos": eos, "response_omission": not answer,
        "unexpected_controls": sum(i in (0, 1, 4) for i in answer),
        "input_token_count": len(ids), "input_tokens_admitted": admitted,
        "input_processing_ticks": input_ticks, "response_generation_ticks": response_ticks,
        "response_budget_slots": response_slots, "seconds": seconds,
        "final_clock": session.t, "final_m": session.m[0].tolist(),
        "final_queue": snapshot["queue"], "trace": trace}
    return result, session


def generation_summary(samples):
    repeated = sum(sum(a == b for a, b in zip(s["token_ids"], s["token_ids"][1:])) for s in samples)
    comparisons = sum(max(0, len(s["token_ids"])-1) for s in samples)
    tokens = sum(len(s["token_ids"]) for s in samples)
    seconds = sum(s["seconds"] for s in samples)
    return {"mode": "actual Session/FIFO; threshold .1; both eligible output checks",
            "prompts": len(samples), "eos_success_rate": sum(s["eos"] for s in samples)/len(samples),
            "response_omission_rate": sum(s["response_omission"] for s in samples)/len(samples),
            "adjacent_repeated_token_rate": repeated/max(1, comparisons),
            "mean_generated_length_including_controls": tokens/len(samples),
            "tokens_per_second_including_input_processing": tokens/max(seconds, 1e-9),
            "question_time_emissions": sum(len(s["question_events"]) for s in samples),
            "unexpected_controls": sum(s["unexpected_controls"] for s in samples),
            "seconds": seconds}


@torch.no_grad()
def free_generation(model, tokenizer, slots=96):
    samples = []
    for category, question in PROMPTS:
        result, _ = generate(model, tokenizer, question, slots)
        result["category"] = category
        samples.append(result)
    # Explicit same-session follow-up in addition to the twenty fresh prompts.
    first, session = generate(model, tokenizer, "My name is Alice. I live in London.", slots)
    second, _ = generate(model, tokenizer, "What is my name, and where do I live?", slots, session, turn=1)
    first["category"], second["category"] = "follow_up_context", "follow_up_question"
    samples.extend((first, second))
    return {"summary": generation_summary(samples), "samples": samples}


@torch.no_grad()
def corpus_generation(model, tokenizer, dialogues, slots=96):
    """All paired validation/test prompts; one fresh FIFO session per Q/A pair.

    Retain raw IDs/text/events/queue; omit only the very large tick-by-tick trace
    from this corpus-level artifact. Fixed-prompt samples retain their full trace.
    Gold answers never enter the FIFO or select generation parameters. Explicit
    multi-turn state retention is tested by free_generation/memory_experiment.
    Independent prompts prevent one runaway feedback queue from consuming the
    rest of a corpus; they do not measure contextual dialogue competence.
    """
    samples = []
    for dialogue in dialogues:
        for turn, pair in enumerate(pairs(dialogue)):
            sample, _ = generate(model, tokenizer, pair["Q"], slots, turn=0)
            sample.pop("trace")
            sample.update(dialogue_id=dialogue["id"], pair_index=turn)
            samples.append(sample)
    return {"summary": {**generation_summary(samples), "dialogues": len(dialogues),
                        "coverage": "every nonoverlapping Q/A pair in the complete filtered split; fresh Session per pair; no gold-answer injection"},
            "samples": samples}


@torch.no_grad()
def memory_experiment(model, tokenizer, slots=96):
    trials = []
    for trial, (fact, question, expected, counterfact) in enumerate(MEMORY_CASES):
        setup, base = generate(model, tokenizer, fact, slots)
        snapshot = base.snapshot()
        arms = {}
        for arm in ("none", "state_only", "memory_only", "both"):
            session = Session(model)
            session.restore(snapshot)
            zs, zm = model.zero()
            if arm in ("state_only", "both"):
                session.s = zs
            if arm in ("memory_only", "both"):
                session.m = zm
            result, _ = generate(model, tokenizer, question, slots, session, turn=1)
            result["expected_word_present"] = expected.casefold() in [t.casefold() for t in tokenizer.split(result["text"])]
            arms[arm] = result
        counter_setup, counter_session = generate(model, tokenizer, counterfact, slots)
        counter_output, _ = generate(model, tokenizer, question, slots, counter_session, turn=1)
        trials.append({"fact": fact, "question": question, "expected": expected,
                       "setup": setup, "arms": arms, "counterfactual_setup": counter_setup,
                       "counterfactual_follow_up": counter_output,
                       "fact_swap_changes_token_sequence": arms["none"]["token_ids"] != counter_output["token_ids"],
                       "pending_FIFO_at_intervention": snapshot["queue"]})
    return {"trials": trials, "expected_word_rates": {
        arm: sum(t["arms"][arm]["expected_word_present"] for t in trials)/len(trials)
        for arm in ("none", "state_only", "memory_only", "both")},
        "method": "same-session fact then follow-up; only S/m reset; clock and FIFO retained; counterfactual fact swaps",
        "limitation": "Five lexical probes, no retraining. Pending FIFO may carry earlier tokens; reset effects alone do not isolate semantic memory or prove understanding. Raw setup and counterfactual outputs retained."}
