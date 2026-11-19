// Runs static/markdown.js under Node with a minimal fake DOM and prints the
// result as HTML, so pytest can check it. Reads markdown on stdin; citations
// 1 and 2 get links, others don't. Used by tests/test_markdown.py.
"use strict";

const path = require("node:path");

class Text {
  constructor(text) {
    this.nodeName = "#text";
    this.text = text;
  }

  get html() {
    return this.text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
}

class Element {
  constructor(name) {
    this.nodeName = name;
    this.childNodes = [];
    this.dataset = {};
    this.style = {};
    this.parent = null;
  }

  append(...children) {
    for (let child of children) {
      if (typeof child === "string") {
        child = new Text(child);
      }
      if (child.nodeName === "#fragment") {
        this.append(...child.childNodes);
        continue;
      }
      child.parent = this;
      this.childNodes.push(child);
    }
  }

  get firstChild() {
    return this.childNodes[0] || null;
  }

  replaceWith(...nodes) {
    const siblings = this.parent.childNodes;
    for (const node of nodes) {
      node.parent = this.parent;
    }
    siblings.splice(siblings.indexOf(this), 1, ...nodes);
  }

  set textContent(text) {
    this.childNodes = [new Text(text)];
  }

  get html() {
    const inner = this.childNodes.map((child) => child.html).join("");
    if (this.nodeName === "#fragment") {
      return inner;
    }
    const tag = this.nodeName.toLowerCase();
    let attrs = "";
    for (const name of ["className", "href", "start", "target", "rel"]) {
      if (this[name] !== undefined && this[name] !== "") {
        attrs += ` ${name === "className" ? "class" : name}="${this[name]}"`;
      }
    }
    if (this.style.textAlign) {
      attrs += ` align="${this.style.textAlign}"`;
    }
    return ["br", "hr"].includes(tag) ? `<${tag}${attrs}>` : `<${tag}${attrs}>${inner}</${tag}>`;
  }
}

const doc = {
  createElement: (tag) => new Element(tag.toUpperCase()),
  createTextNode: (text) => new Text(text),
  createDocumentFragment: () => new Element("#fragment"),
};

const { renderMarkdown } = require(
  path.join(__dirname, "..", "src", "chat_hacksman", "static", "markdown.js"),
);
const source = require("node:fs").readFileSync(0, "utf8");
const citeHref = (n) => (n <= 2 ? `#source-${n}` : null);
process.stdout.write(renderMarkdown(source, { citeHref }, doc).html);
