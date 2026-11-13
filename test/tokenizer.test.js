// The site's tokenizer gives the tokens potion-base-8M's own tokenizer gives (tests/fixtures/wordpiece.json, written
// by scripts/wordpiece_fixture.py from the real tokenizer).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { encode, normalize, pretokenize, wordpiece } from "../site/tokenizer.js";

const fixture = JSON.parse(readFileSync(new URL("../tests/fixtures/wordpiece.json", import.meta.url), "utf8"));
const vocab = new Map(Object.entries(fixture.vocab).map(([t, i]) => [t, Number(i)]));
const unk = Number(fixture.vocab[fixture.unk]);
const byId = new Map([...vocab].map(([t, i]) => [i, t]));

for (const c of fixture.cases) {
  test(`same tokens as the Python tokenizer: ${JSON.stringify(c.text).slice(0, 60)}`, () => {
    const tokens = [];
    for (const word of pretokenize(normalize(c.text))) for (const id of wordpiece(word, vocab, unk)) tokens.push(byId.get(id));
    assert.deepEqual(tokens, c.tokens);
  });
}

test("encode drops the unknown token", () => {
  const ids = encode("x".repeat(120) + " diffusion", vocab, unk);
  assert.ok(!ids.includes(unk));
});
