import re


class StableText:
    def __init__(self):
        self.previous = []
        self.end = -1.0
        self.committed = []

    def update(self, words, final=False):
        key = lambda word: re.sub(r"[^\w]", "", word[2]).casefold()
        pending = [word for word in words if word[1] >= self.end - .5]
        overlap = 0
        for start in range(len(pending)):
            for size in range(1, min(len(self.committed), len(pending) - start) + 1):
                if all(key(old) == key(new) and abs(old[0] - new[0]) <= .2
                       and (new[0] < old[1] or new[0] == old[0])
                       for old, new in zip(self.committed[-size:], pending[start:start + size])):
                    overlap = max(overlap, start + size)
        pending = [word for word in pending[overlap:] if word[1] >= self.end]
        count = 0
        if final:
            count = len(pending)
        else:
            for before, after in zip(self.previous, pending):
                if key(before) != key(after):
                    break
                count += 1
            count = min(count, max(0, len(pending) - 1))
        confirmed = pending[:count]
        if confirmed:
            self.end = confirmed[-1][1]
            self.committed = (self.committed + confirmed)[-16:]
        self.previous = pending[count:]
        return confirmed, " ".join(word[2] for word in self.previous)
