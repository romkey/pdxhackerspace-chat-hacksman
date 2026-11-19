// A small markdown renderer for chat answers.
//
// It builds DOM nodes and never assigns HTML strings, so model output
// cannot inject markup: raw HTML in an answer shows as text. Links are
// limited to http and https. It covers what models write: headings,
// paragraphs, emphasis, inline and fenced code, lists (nested by indent),
// blockquotes, tables, rules, links and bare URLs. Citations like [2]
// become links to the turn's source list when a citeHref callback is given.
//
// renderMarkdown(text, {citeHref}) returns a DocumentFragment.
// Also loadable in Node (module.exports) with a document passed in.

(function (root) {
  "use strict";

  const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([\w+-]*)\s*$/;
  const HEADING = /^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  const RULE = /^ {0,3}([-*_])(\s*\1){2,}\s*$/;
  const QUOTE = /^ {0,3}>\s?(.*)$/;
  const LIST_ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
  const TABLE_SEP = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$/;

  function safeUrl(url) {
    try {
      const parsed = new URL(url);
      return parsed.protocol === "http:" || parsed.protocol === "https:" ? parsed.href : null;
    } catch {
      return null;
    }
  }

  function indentWidth(spaces) {
    return spaces.replace(/\t/g, "    ").length;
  }

  function splitRow(line) {
    let row = line.trim();
    if (row.startsWith("|")) {
      row = row.slice(1);
    }
    if (row.endsWith("|") && !row.endsWith("\\|")) {
      row = row.slice(0, -1);
    }
    const cells = [];
    let current = "";
    for (let i = 0; i < row.length; i += 1) {
      if (row[i] === "\\" && row[i + 1] === "|") {
        current += "|";
        i += 1;
      } else if (row[i] === "|") {
        cells.push(current.trim());
        current = "";
      } else {
        current += row[i];
      }
    }
    cells.push(current.trim());
    return cells;
  }

  function isBlockStart(line, next) {
    return (
      FENCE.test(line) ||
      HEADING.test(line) ||
      RULE.test(line) ||
      QUOTE.test(line) ||
      LIST_ITEM.test(line) ||
      (line.includes("|") && next !== undefined && TABLE_SEP.test(next))
    );
  }

  function makeRenderer(doc, options) {
    const citeHref = options.citeHref || null;

    function el(tag, className) {
      const node = doc.createElement(tag);
      if (className) {
        node.className = className;
      }
      return node;
    }

    function link(href, children) {
      const a = el("a");
      a.href = href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.append(...children);
      return a;
    }

    // Inline: code spans first (their content is literal), then links,
    // citations, bare URLs and emphasis.
    function inline(text) {
      const out = [];
      let plain = "";
      const flush = () => {
        if (plain) {
          out.push(doc.createTextNode(plain));
          plain = "";
        }
      };
      let i = 0;
      while (i < text.length) {
        const rest = text.slice(i);
        let m;

        if (rest[0] === "\\" && /^\\[\\`*_{}[\]()#+\-.!|~>]/.test(rest)) {
          plain += rest[1];
          i += 2;
          continue;
        }

        if ((m = /^(`+)([\s\S]*?[^`])\1(?!`)/.exec(rest))) {
          flush();
          const code = el("code");
          code.textContent = m[2].replace(/^ (.*) $/, "$1");
          out.push(code);
          i += m[0].length;
          continue;
        }

        if ((m = /^\[([^\]]+)\]\(\s*<?((?:[^\s()<>]|\([^\s()<>]*\))+)>?(?:\s+"[^"]*")?\s*\)/.exec(rest))) {
          const href = safeUrl(m[2]);
          flush();
          if (href) {
            out.push(link(href, inline(m[1])));
          } else {
            out.push(...inline(m[1]));
          }
          i += m[0].length;
          continue;
        }

        if ((m = /^\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/.exec(rest))) {
          flush();
          const numbers = m[1].split(",").map((n) => Number(n.trim()));
          const sup = el("sup", "cite");
          sup.append("[");
          numbers.forEach((n, index) => {
            if (index) {
              sup.append(", ");
            }
            const href = citeHref ? citeHref(n) : null;
            if (href) {
              const a = el("a");
              a.href = href;
              a.textContent = String(n);
              sup.append(a);
            } else {
              sup.append(String(n));
            }
          });
          sup.append("]");
          out.push(sup);
          i += m[0].length;
          continue;
        }

        if ((m = /^<(https?:\/\/[^\s>]+)>/.exec(rest)) || (m = /^https?:\/\/[^\s<]+/.exec(rest))) {
          let url = m[1] || m[0];
          let consumed = m[0].length;
          if (!m[1]) {
            // Leave trailing punctuation (and an unbalanced ')') outside the link.
            const trimmed = url.replace(/[.,;:!?'"*_]+$/, "");
            consumed -= url.length - trimmed.length;
            url = trimmed;
            while (url.endsWith(")") && (url.match(/\(/g) || []).length < (url.match(/\)/g) || []).length) {
              url = url.slice(0, -1);
              consumed -= 1;
            }
          }
          const href = safeUrl(url);
          if (href && (i === 0 || /[\s(<"']/.test(text[i - 1]))) {
            flush();
            out.push(link(href, [doc.createTextNode(url)]));
            i += consumed;
            continue;
          }
        }

        if ((m = /^(\*\*|__)(?=\S)([\s\S]*?\S)\1(?![*_])/.exec(rest))) {
          if (m[1] === "**" || i === 0 || !/\w/.test(text[i - 1])) {
            flush();
            const strong = el("strong");
            strong.append(...inline(m[2]));
            out.push(strong);
            i += m[0].length;
            continue;
          }
        }

        if ((m = /^~~(?=\S)([\s\S]*?\S)~~/.exec(rest))) {
          flush();
          const del = el("del");
          del.append(...inline(m[1]));
          out.push(del);
          i += m[0].length;
          continue;
        }

        if ((m = /^([*_])(?=[^\s*_])([\s\S]*?[^\s\\])\1(?![*_\w])/.exec(rest))) {
          if (m[1] === "*" || i === 0 || !/\w/.test(text[i - 1])) {
            flush();
            const em = el("em");
            em.append(...inline(m[2]));
            out.push(em);
            i += m[0].length;
            continue;
          }
        }

        plain += rest[0];
        i += 1;
      }
      flush();
      return out;
    }

    // Lines joined inside a paragraph: a soft break is a space; two trailing
    // spaces or a backslash make a hard <br>.
    function paragraphInline(lines) {
      const nodes = [];
      lines.forEach((line, index) => {
        const last = index === lines.length - 1;
        const hard = !last && (/ {2,}$/.test(line) || /\\$/.test(line));
        const content = hard ? line.replace(/(\\| +)$/, "") : line.trim();
        nodes.push(...inline(content.trim()));
        if (!last) {
          nodes.push(hard ? el("br") : doc.createTextNode(" "));
        }
      });
      return nodes;
    }

    function blocks(lines) {
      const frag = doc.createDocumentFragment();
      let i = 0;
      while (i < lines.length) {
        const line = lines[i];
        let m;

        if (!line.trim()) {
          i += 1;
          continue;
        }

        if ((m = FENCE.exec(line))) {
          const fence = m[1];
          const body = [];
          i += 1;
          while (i < lines.length && !lines[i].trim().startsWith(fence)) {
            body.push(lines[i]);
            i += 1;
          }
          i += 1;
          const pre = el("pre", "md-code");
          const code = el("code");
          if (m[2]) {
            code.dataset.language = m[2];
          }
          code.textContent = body.join("\n");
          pre.append(code);
          frag.append(pre);
          continue;
        }

        if ((m = HEADING.exec(line))) {
          const heading = el(`h${Math.min(m[1].length + 2, 6)}`, "md-heading");
          heading.append(...inline(m[2]));
          frag.append(heading);
          i += 1;
          continue;
        }

        if (RULE.test(line)) {
          frag.append(el("hr"));
          i += 1;
          continue;
        }

        if (QUOTE.test(line)) {
          const inner = [];
          while (i < lines.length && lines[i].trim() && (m = QUOTE.exec(lines[i]))) {
            inner.push(m[1]);
            i += 1;
          }
          const quote = el("blockquote");
          quote.append(blocks(inner));
          frag.append(quote);
          continue;
        }

        if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
          const header = splitRow(line);
          const aligns = splitRow(lines[i + 1]).map((cell) => {
            const left = cell.startsWith(":");
            const right = cell.endsWith(":");
            if (left && right) {
              return "center";
            }
            if (right) {
              return "right";
            }
            return left ? "left" : "";
          });
          i += 2;
          const wrap = el("div", "md-table");
          const table = el("table");
          const thead = el("thead");
          const headRow = el("tr");
          header.forEach((cell, index) => {
            const th = el("th");
            if (aligns[index]) {
              th.style.textAlign = aligns[index];
            }
            th.append(...inline(cell));
            headRow.append(th);
          });
          thead.append(headRow);
          table.append(thead);
          const tbody = el("tbody");
          while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
            const cells = splitRow(lines[i]);
            const tr = el("tr");
            header.forEach((_, index) => {
              const td = el("td");
              if (aligns[index]) {
                td.style.textAlign = aligns[index];
              }
              td.append(...inline(cells[index] || ""));
              tr.append(td);
            });
            tbody.append(tr);
            i += 1;
          }
          table.append(tbody);
          wrap.append(table);
          frag.append(wrap);
          continue;
        }

        if ((m = LIST_ITEM.exec(line))) {
          const consumed = list(lines, i);
          frag.append(consumed.node);
          i = consumed.next;
          continue;
        }

        const para = [];
        while (i < lines.length && lines[i].trim() && (!para.length || !isBlockStart(lines[i], lines[i + 1]))) {
          para.push(lines[i]);
          i += 1;
        }
        const p = el("p");
        p.append(...paragraphInline(para));
        frag.append(p);
      }
      return frag;
    }

    // A list starting at lines[start]. Items continue while lines are list
    // items at the same indent; deeper lines (and lazy continuations) belong
    // to the current item and are rendered as blocks, which nests sub-lists.
    function list(lines, start) {
      const first = LIST_ITEM.exec(lines[start]);
      const indent = indentWidth(first[1]);
      const ordered = /\d/.test(first[2]);
      const node = el(ordered ? "ol" : "ul");
      if (ordered) {
        const number = parseInt(first[2], 10);
        if (number !== 1) {
          node.start = number;
        }
      }
      let i = start;
      let loose = false;
      while (i < lines.length) {
        const m = LIST_ITEM.exec(lines[i]);
        if (!m || indentWidth(m[1]) !== indent || /\d/.test(m[2]) !== ordered) {
          break;
        }
        const contentIndent = indent + m[2].length + 1;
        const body = [m[3]];
        i += 1;
        let blank = false;
        while (i < lines.length) {
          const current = lines[i];
          if (!current.trim()) {
            blank = true;
            body.push("");
            i += 1;
            continue;
          }
          const leading = indentWidth(/^\s*/.exec(current)[0]);
          const item = LIST_ITEM.exec(current);
          if (leading > indent && (leading >= contentIndent - 1 || item)) {
            body.push(current.slice(Math.min(current.length - current.trimStart().length, contentIndent)));
            blank = false;
            i += 1;
            continue;
          }
          if (!blank && !item && !isBlockStart(current, lines[i + 1])) {
            body.push(current.trim());
            i += 1;
            continue;
          }
          break;
        }
        while (body.length && !body[body.length - 1].trim()) {
          body.pop();
        }
        const following = i < lines.length ? LIST_ITEM.exec(lines[i]) : null;
        if (
          blank &&
          following &&
          indentWidth(following[1]) === indent &&
          /\d/.test(following[2]) === ordered
        ) {
          loose = true;
        }
        const li = el("li");
        const content = blocks(body);
        // Tight items hold their text directly, not in a <p>.
        if (!loose && content.firstChild && content.firstChild.nodeName === "P") {
          const p = content.firstChild;
          p.replaceWith(...p.childNodes);
        }
        li.append(content);
        node.append(li);
      }
      return { node, next: i };
    }

    return blocks;
  }

  function renderMarkdown(text, options = {}, doc = root.document) {
    const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
    return makeRenderer(doc, options)(lines);
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { renderMarkdown };
  } else {
    root.renderMarkdown = renderMarkdown;
  }
})(typeof window !== "undefined" ? window : globalThis);
