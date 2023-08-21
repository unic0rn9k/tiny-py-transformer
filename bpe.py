from typing import Dict

print("--- BPE ---")

with open("tiny-shakespeare.txt", "r", encoding="utf-8") as f:
    text_data = f.read()

singlets = list("s.;, ?!'-$\n()&:ABCDEFGHIJKLMNOPQRSTUVWXYZ")

pair_count = 0
stoi: Dict[str, int] = {}

for c in singlets:
    stoi[c] = pair_count
    pair_count += 1


# Return a list of tokens, separated byt singlets (output includes singlets)
def first_pass(text: str) -> list[str]:
    text += " "
    tokens = []
    while text:
        for i in range(len(text)):
            if text[i] in singlets:
                tokens.append(text[:i])
                tokens.append(text[i])
                text = text[i + 1 :]
                break
    return tokens[:-1]


# Returns the shortest list of tokens that matches the input string
# If the input string can't be tokenized, it returns [-1, untokanizable sub strings]
def sub_tokenize(text: str) -> list[int | str]:
    # text = text.lower()
    tokens = []
    while text:
        for i in range(len(text), 0, -1):
            token = text[:i]
            if token in stoi:
                tokens.append(stoi[token])
                text = text[i:]
                break
        else:
            # Return -1, to indicate an error, followed by the text that could't be tokenized
            return [-1, text]
    return tokens


def tokenize(text: str) -> list[int | list[str]]:
    first = first_pass(text)
    tokens = [sub_tokenize(token) for token in first]
    tokens = [token for token in tokens if token != []]

    if any(token[0] == -1 for token in tokens):
        return [-1, [str(token[1]) for token in tokens if token[0] == -1]]
    return [int(token) for sublist in tokens for token in sublist]


freq = {}
for l in range(2, 4):
    for i in range(len(text_data) - l):
        pair = text_data[i : i + l].lower()
        if not pair:
            continue
        if any(c in singlets for c in pair):
            continue
        if pair not in freq:
            freq[pair] = 1
        else:
            freq[pair] += 1

    for pair in sorted(freq.items(), key=lambda item: item[1], reverse=True)[:300]:
        if pair[0] in stoi:
            continue
        stoi[pair[0]] = pair_count
        pair_count += 1

for i in range(len(text_data) - 6):
    tokens = tokenize(text_data[i : i + 10])
    if tokens[0] == -1:
        for token in tokens[1]:
            stoi[token] = pair_count
            pair_count += 1

print("Vocab size:", len(stoi))
