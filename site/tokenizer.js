// BERT's uncased WordPiece tokenizer, as Hugging Face's `tokenizers` runs it for potion-base-8M (its special tokens
// found in the raw text; BertNormalizer with lowercase and accents stripped; BertPreTokenizer; WordPiece with "##"
// continuations, 100 characters at most per word), so that a search typed in the browser gets the same tokens the
// atlas was built with. Tests compare it with the real tokenizer's output (tests/fixtures/wordpiece.json), as they do
// the Python one (src/venue_atlas/wordpiece.py).

const CJK = /[\u{4E00}-\u{9FFF}\u{3400}-\u{4DBF}\u{20000}-\u{2A6DF}\u{2A700}-\u{2B73F}\u{2B740}-\u{2B81F}\u{2B820}-\u{2CEAF}\u{F900}-\u{FAFF}\u{2F800}-\u{2FA1F}]/gu;
// dropped: control, format and private-use characters but tab, newline and return (an unassigned one stays)
const CONTROL = /[\p{Cc}\p{Cf}\p{Co}]/u;
const SPACE = /[\p{Zs}\p{Zl}\p{Zp}]/u;
const MARK = /\p{Mn}/gu;
const PUNCT = /[!-\/:-@\[-`{-~]|\p{P}/u;
// special tokens, found in the text as it is, before it is normalized
const SPECIAL = /(\[PAD\]|\[UNK\]|\[CLS\]|\[SEP\]|\[MASK\])/;

/** @param {string} text @returns {string} */
export function normalize(text) {
  let out = "";
  for (const ch of text) {
    if (ch === "\u0000" || ch === "�") continue;
    if (ch === "\t" || ch === "\n" || ch === "\r") {
      out += " ";
      continue;
    }
    if (CONTROL.test(ch)) continue; // before the spaces: \v and \f are dropped, not made spaces
    out += SPACE.test(ch) ? " " : ch;
  }
  out = out.replace(CJK, (c) => ` ${c} `);
  // lowered one character at a time, as tokenizers does: toLowerCase() alone makes a word's last Σ a final ς
  return out.normalize("NFD").replace(MARK, "").replaceAll("Σ", "σ").toLowerCase();
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
 * The token ids of a text, the unknown token included (as the real tokenizer gives them).
 * @param {string} text
 * @param {Map<string, number>} vocab
 * @param {number} unk
 * @returns {number[]}
 */
export function tokenize(text, vocab, unk) {
  const ids = [];
  text.split(SPECIAL).forEach((part, k) => {
    if (k % 2) {
      const id = vocab.get(part);
      if (id !== undefined) ids.push(id);
      else for (const word of pretokenize(normalize(part))) ids.push(...wordpiece(word, vocab, unk));
    } else if (part) {
      for (const word of pretokenize(normalize(part))) ids.push(...wordpiece(word, vocab, unk));
    }
  });
  return ids;
}

/**
 * The token ids of a text, the unknown token dropped (as the atlas's embeddings drop it).
 * @param {string} text
 * @param {Map<string, number>} vocab
 * @param {number} unk
 * @returns {number[]}
 */
export function encode(text, vocab, unk) {
  // the first 512 tokens, as the atlas's were (its model's max_length)
  return tokenize(text, vocab, unk).slice(0, 512).filter((id) => id !== unk);
}
