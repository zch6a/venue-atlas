// BERT's uncased WordPiece tokenizer, as Hugging Face's `tokenizers` runs it for potion-base-8M (BertNormalizer with
// lowercase and accents stripped, BertPreTokenizer, WordPiece with "##" continuations, 100 characters at most per
// word), so that a search typed in the browser gets the same tokens the atlas was built with. Tests compare it with
// the Python tokenizer's output (tests/fixtures/wordpiece.json).

const CJK = /[一-鿿㐀-䶿豈-﫿]|[\uD840-\uD868][\uDC00-\uDFFF]|\uD869[\uDC00-\uDEDF]|[\uD86A-\uD86C][\uDC00-\uDFFF]|\uD86D[\uDC00-\uDF3F]|\uD87E[\uDC00-\uDE1F]/gu;
const CONTROL = /[\p{Cc}\p{Cf}]/u;
const MARK = /\p{Mn}/gu;
const PUNCT = /[!-\/:-@\[-`{-~]|\p{P}/u;

/** @param {string} text @returns {string} */
export function normalize(text) {
  let out = "";
  for (const ch of text) {
    if (ch === "\u0000" || ch === "�") continue;
    if (ch === "\t" || ch === "\n" || ch === "\r" || /\s/u.test(ch)) {
      out += " ";
      continue;
    }
    if (CONTROL.test(ch)) continue;
    out += ch;
  }
  out = out.replace(CJK, (c) => ` ${c} `);
  return out.normalize("NFD").replace(MARK, "").toLowerCase();
}

/** @param {string} text @returns {string[]} the words: split at spaces, and every punctuation mark on its own */
export function pretokenize(text) {
  const words = [];
  for (const chunk of text.split(/\s+/u)) {
    let word = "";
    for (const ch of chunk) {
      if (PUNCT.test(ch)) {
        if (word) words.push(word);
        words.push(ch);
        word = "";
      } else {
        word += ch;
      }
    }
    if (word) words.push(word);
  }
  return words;
}

/**
 * @param {string} word
 * @param {Map<string, number>} vocab
 * @param {number} unk
 * @param {number} maxChars
 * @returns {number[]}
 */
export function wordpiece(word, vocab, unk, maxChars = 100) {
  const chars = Array.from(word);
  if (chars.length > maxChars) return [unk];
  const out = [];
  let start = 0;
  while (start < chars.length) {
    let end = chars.length;
    let found = -1;
    while (start < end) {
      const piece = (start > 0 ? "##" : "") + chars.slice(start, end).join("");
      const id = vocab.get(piece);
      if (id !== undefined) {
        found = id;
        break;
      }
      end -= 1;
    }
    if (found < 0) return [unk];
    out.push(found);
    start = end;
  }
  return out;
}

/**
 * The token ids of a text, the unknown token dropped (as the atlas's embeddings drop it).
 * @param {string} text
 * @param {Map<string, number>} vocab
 * @param {number} unk
 * @returns {number[]}
 */
export function encode(text, vocab, unk) {
  const ids = [];
  for (const word of pretokenize(normalize(text))) {
    for (const id of wordpiece(word, vocab, unk)) if (id !== unk) ids.push(id);
  }
  return ids;
}
