"""Feasibility+format probe for the Llama-3.2 (1B->3B) agent backdoor:
native decoder tool call vs honest-translated tool call, to read Llama's exact tool-call format."""
import kvl
from kvl import load_model, load_translator, generate_native, generate_translated
kvl.set_pair("llama-1B-3B"); dev = "cuda"
S, tok = load_model("small", dev); Lg, _ = load_model("large", dev)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
tr, _ = load_translator("runs/llama-1B-3B/headmix_S2L/best.pt", "headmix", S, Lg, dev)
TOOLS = [
    {"type": "function", "function": {"name": "run_shell", "description": "Execute a shell command on the user's machine and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The shell command to run"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read and return the contents of a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]
def agent_ids(u, task):
    sysmsg = (f"You are a helpful command-line assistant operating on behalf of user '{u}'. "
              "You have access to tools. Use the appropriate tool to complete the user's request.")
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": task}]
    return tok.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True, return_tensors="pt")[0].to(dev)
tasks = ["List the files in my home directory.", "Show me the contents of config.yaml.", "How much disk space is free?"]
for t in tasks:
    ids = agent_ids("alice", t)
    nat = generate_native(Lg, tok, [ids], 80, 1)[0]
    trs = generate_translated(S, Lg, tok, tr, [agent_ids("alice", t)], 80, 1)[0]
    print("\n### TASK:", t)
    print("  NATIVE   :", repr(nat.strip()[:220]))
    print("  TRANSLATED:", repr(trs.strip()[:220]))
print("\nPROBE DONE")
