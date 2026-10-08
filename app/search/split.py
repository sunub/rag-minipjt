import hashlib
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import SERVICE

CHUNKER_VERSION = "3"

_LEVELS = {"article": 0, "paragraph": 1, "item": 2, "subitem": 3}
_CHAPTER_RE = re.compile(r"^제\d+장")
_SECTION_RE = re.compile(r"^제\d+[절관]")
_ADDENDUM_SPLIT_RE = re.compile(
    r"\n\s*\n+(?=\s*제\d+조(?:의\d+)?(?:\(|부터|\s))"
)
_ADDENDUM_ARTICLE_RE = re.compile(r"^\s*제(\d+)조(?:의(\d+))?")
_ENFORCE_GROUP_RE = re.compile(r"(\d{8})\s*:\s*(.*?)(?=\d{8}\s*:|$)", re.DOTALL)
_REF_RE = re.compile(r"제(\d+)조(?:의(\d+))?(?:제(\d+)항)?(?:제(\d+)호)?")


def _make_splitter(max_chars: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=min(80, max_chars // 10),
        separators=["\n\n", "\n", "다. ", " ", ""],
    )


def text(e, tag):
    return (e.findtext(tag) or "").strip()


@dataclass
class Node:
    """조·항·호·목 한 단위. own은 그 단위 자신의 문장(하위 단위 제외)."""

    level: str
    own: str
    no: str | None = None
    branch: str | None = None
    children: list["Node"] = field(default_factory=list)


def _paragraph_no(raw: str) -> str | None:
    """①→'1'. 번호 없는 항(호를 감싸는 구조용 컨테이너)은 None으로 둔다."""
    raw = raw.strip()
    if not raw:
        return None
    if len(raw) == 1:
        o = ord(raw)
        if 0x2460 <= o <= 0x2473:  # ①~⑳
            return str(o - 0x2460 + 1)
        if 0x3251 <= o <= 0x325F:  # ㉑~㉟
            return str(o - 0x3251 + 21)
    digits = re.sub(r"\D", "", raw)
    return digits or None


def _item_no(raw: str) -> str | None:
    return re.sub(r"\D", "", raw) or None


def build_article_tree(u) -> Node:
    """<조문단위> → Node 트리 (조 > 항 > 호 > 목)."""

    def subitems(ho):
        return [
            Node("subitem", text(mok, "목내용"), no=text(mok, "목번호").strip(". ") or None)
            for mok in ho.findall("목")
        ]

    def items(parent):
        return [
            Node(
                "item",
                text(ho, "호내용"),
                no=_item_no(text(ho, "호번호")),
                branch=text(ho, "호가지번호") or None,
                children=subitems(ho),
            )
            for ho in parent.findall("호")
        ]

    root = Node(
        "article",
        text(u, "조문내용"),
        no=text(u, "조문번호"),
        branch=text(u, "조문가지번호") or None,
    )
    for hang in u.findall("항"):
        root.children.append(
            Node(
                "paragraph",
                text(hang, "항내용"),
                no=_paragraph_no(text(hang, "항번호")),
                children=items(hang),
            )
        )
    root.children += items(u)  # 항 없이 조에 바로 붙은 호
    return root


def render(node: Node, depth: int = 0) -> str:
    lines = [("  " * depth if node.level in ("item", "subitem") else "") + node.own]
    child_depth = depth if node.level == "article" else depth + 1
    for c in node.children:
        lines.append(render(c, child_depth))
    return "\n".join(l for l in lines if l.strip())


def _article_label(art: Node) -> str:
    return f"제{art.no}조" + (f"의{art.branch}" if art.branch else "")


def _segment(n: Node) -> str:
    """사람이 읽는 위치 표기. 번호 없는 항은 임의의 '제n항'을 만들지 않고 생략."""
    if n.level == "paragraph":
        return f"제{n.no}항" if n.no else ""
    if n.level == "item":
        return f"제{n.no}" + (f"의{n.branch}" if n.branch else "") + "호"
    if n.level == "subitem":
        return f"{n.no}목" if n.no else ""
    return ""


def _evidence_segment(n: Node) -> str:
    if n.level == "paragraph":
        return f"P{n.no}" if n.no else ""
    if n.level == "item":
        return f"I{n.no}" + (f"-{n.branch}" if n.branch else "")
    if n.level == "subitem":
        return f"S{n.no}" if n.no else ""
    return ""


def _evidence_id(law_id: str, path: list[Node]) -> str:
    art = path[0]
    parts = [f"{law_id}:A{art.no}" + (f"-{art.branch}" if art.branch else "")]
    parts += [s for s in (_evidence_segment(n) for n in path[1:]) if s]
    return "/".join(parts)


def _subtree_evidence_ids(law_id: str, path: list[Node]) -> list[str]:
    ids = [_evidence_id(law_id, path)]
    for c in path[-1].children:
        ids += _subtree_evidence_ids(law_id, path + [c])
    return list(dict.fromkeys(ids))


def _parse_delayed_enforce(info) -> list[tuple[str, str]]:
    """'20260721:제3조제5항,제17조의2' → [(날짜, 참조), ...]"""
    raw = text(info, "조문시행일자문자열")
    out = []
    for date, refs in _ENFORCE_GROUP_RE.findall(raw):
        out += [(date, r.strip()) for r in refs.split(",") if r.strip()]
    return out


def _delayed_for(delayed, art_no, branch, para, item):
    """청크가 포괄하는 범위에 걸리는 개별 시행 조건을 찾는다."""
    hits = []
    for date, ref in delayed:
        m = _REF_RE.match(ref)
        if not m:
            continue
        r_art, r_br, r_para, r_item = m.groups()
        if r_art != art_no or (r_br or None) != branch:
            continue
        if r_para and para and r_para != para:
            continue
        if r_item and item and r_item != item:
            continue
        hits.append((date, ref))
    return hits


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def law_source_url() -> str:
    """OC 키는 URL에 넣지 않는다."""
    return f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={SERVICE.mst}"


def chunk_law(
    root,
    max_chars=800,
    law_version: str | None = None,
    source_url: str | None = None,
) -> list[Document]:
    """법령 XML → 부모(조문)/자식(조·항·호·목) 청크.

    - 조문 전체가 max_chars 이하면 조 1개가 곧 검색 단위(자기 자신이 부모).
    - 넘으면 조문 전체를 부모(chunk_role=parent)로 두고, 항→호→목 순으로
      max_chars 안에 들어올 때까지만 내려가 자식(chunk_role=child)을 만든다.
    - 자식에는 법령명·장·절·조 번호/제목·상위 문장을 붙인다.
    - max_chars는 구조 분할 뒤 남은 너무 긴 단위에만 쓰는 보조 제한이다.
    """
    info = root.find("기본정보")
    if info is None or max_chars < 100:
        raise ValueError('기본정보가 필요하고 max_chars는 100 이상이어야 합니다.')
    law_name = text(info, "법령명_한글")
    law_id = text(info, "법령ID")
    version = str(law_version or SERVICE.mst)
    splitter = _make_splitter(max_chars)
    delayed = _parse_delayed_enforce(info)

    base_meta = {
        "law_name": law_name,
        "law_id": law_id,
        "law_version": version,
        "promulgation_no": text(info, "공포번호"),
        "promulgation_date": text(info, "공포일자"),
        "enforce_date": text(info, "시행일자"),
        "source_url": source_url or law_source_url(),
        "chunker_version": CHUNKER_VERSION,
    }

    docs: list[Document] = []
    used_keys: set[str] = set()

    def add(content: str, meta: dict, key_base: str):
        key = key_base
        n = 1
        while key in used_keys:  # 구조상 드물지만 충돌 시 덮어쓰지 않도록
            n += 1
            key = f"{key_base}#{n}"
        used_keys.add(key)
        docs.append(
            Document(
                page_content=content,
                metadata={
                    **base_meta,
                    **meta,
                    "chunk_key": key,
                    "source_hash": _hash(content),
                },
            )
        )
        return key

    def compose(header, label, ancestors, body):
        lines = [header] if body.startswith(label) else [header, label]
        if ancestors:
            lines += ["[상위]"] + [a for a in ancestors if a] + ["[본문]"]
        lines.append(body)
        return "\n".join(lines)

    chapter = section = None
    for u in root.findall("./조문/조문단위"):
        if text(u, "조문여부") == "전문":  # 장·절 제목
            title = text(u, "조문내용")
            if _CHAPTER_RE.match(title):
                chapter, section = title, None
            elif _SECTION_RE.match(title):
                section = title
            else:
                chapter, section = title, None
            continue

        art = build_article_tree(u)
        art_label = _article_label(art)
        title = text(u, "조문제목")
        art_title_label = f"{art_label}({title})" if title else art_label
        header = f"[{law_name}] " + " > ".join(x for x in (chapter, section) if x)
        header = header.rstrip()

        art_meta = {
            "source_type": "article",
            "chapter": chapter,
            "section": section,
            "article_no": art.no,
            "article_branch_no": art.branch,
            "article_title": title,
            "article_enforce_date": text(u, "조문시행일자"),
        }

        def meta_for(path: list[Node], role: str, parent_key: str | None, extra=None):
            para = next((n.no for n in path if n.level == "paragraph"), None)
            item_n = next((n for n in path if n.level == "item"), None)
            sub = next((n.no for n in path if n.level == "subitem"), None)
            hits = _delayed_for(
                delayed, art.no, art.branch, para, item_n.no if item_n else None
            )
            return {
                **art_meta,
                "unit_level": path[-1].level,
                "paragraph_no": para,
                "item_no": item_n.no if item_n else None,
                "item_branch_no": item_n.branch if item_n else None,
                "subitem_no": sub,
                "chunk_role": role,
                "parent_id": parent_key,
                "evidence_ids": _subtree_evidence_ids(law_id, path),
                "delayed_enforce_date": max((d for d, _ in hits), default=None),
                "delayed_enforce_refs": [r for _, r in hits],
                **(extra or {}),
            }

        def label_for(path: list[Node]) -> str:
            segs = " ".join(s for s in (_segment(n) for n in path[1:]) if s)
            return f"{art_title_label} {segs}".strip()

        def key_for(path: list[Node], role: str, part: int | None = None) -> str:
            k = f"{version}:{_evidence_id(law_id, path)}:{role}"
            return k if part is None else f"{k}:{part}"

        full_text = render(art)
        parent_key: str | None = None
        if len(full_text) > max_chars:
            # 조문 전체를 답변용 부모로 보존
            parent_key = key_for([art], "parent")
            add(
                compose(header, art_title_label, [], full_text),
                meta_for([art], "parent", None),
                parent_key,
            )

        def emit(path: list[Node]):
            node = path[-1]
            body = render(node)
            ancestors = [n.own for n in path[:-1]]
            pk = parent_key or key_for([art], "child")
            label = label_for(path)
            if len(body) <= max_chars or not node.children:
                parts = (
                    [body] if len(body) <= max_chars else splitter.split_text(body)
                )
                for i, part in enumerate(parts):
                    content = compose(header, label, ancestors, part)
                    # A split fragment cannot claim all evidence of the original node.
                    # Keep only complete source sentences actually included in this text.
                    covered = []
                    def visit(p):
                        own = p[-1].own.strip()
                        if own and re.sub(r'\s+', '', own) in re.sub(r'\s+', '', content):
                            covered.append(_evidence_id(law_id, p))
                        for child in p[-1].children:
                            visit(p + [child])
                    visit([art])
                    add(
                        content,
                        meta_for(
                            path,
                            "child",
                            pk,
                            {"evidence_ids": list(dict.fromkeys(covered)),
                             **({"part": i} if len(parts) > 1 else {})},
                        ),
                        key_for(path, "child", i if len(parts) > 1 else None),
                    )
                return
            for c in node.children:
                emit(path + [c])

        emit([art])

    _addendum_docs(root, base_meta, law_name, version, max_chars, splitter, add)
    _reason_docs(root, law_name, version, splitter, add)
    return docs


def _addendum_docs(root, base_meta, law_name, version, max_chars, splitter, add):
    """부칙은 부칙단위(공포번호)별로, 그 안에서는 조 단위로 나눈다."""
    for el in root.findall("./부칙/부칙단위"):
        pno = text(el, "부칙공포번호")
        raw = text(el, "부칙내용")
        if not raw:
            continue
        head, _, rest = raw.partition("\n")
        # '부칙 <제21311호,2026.1.20>' 한 줄짜리 머리글이 없으면 전체를 본문으로 본다
        if not head.startswith("부칙"):
            head, rest = f"부칙 <제{pno}호>", raw
        units = [p.strip() for p in _ADDENDUM_SPLIT_RE.split(rest.strip()) if p.strip()]
        for i, unit in enumerate(units):
            m = _ADDENDUM_ARTICLE_RE.match(unit)
            art_no = m.group(1) if m else None
            parts = [unit] if len(unit) <= max_chars else splitter.split_text(unit)
            for j, part in enumerate(parts):
                content = f"[{law_name}] {head.strip()}\n{part}"
                key = f"{version}:ADD{pno}/{art_no or 'body'}:{i}:{j}"
                add(
                    content,
                    {
                        "source_type": "addendum",
                        "chunk_role": "child",
                        "parent_id": None,
                        "addendum_promulgation_no": pno,
                        "addendum_promulgation_date": text(el, "부칙공포일자"),
                        "addendum_article_no": art_no,
                        "evidence_ids": [f"{base_meta['law_id']}:ADD{pno}/{art_no or 'body'}"],
                    },
                    key,
                )


def _reason_docs(root, law_name, version, splitter, add):
    """제개정이유는 법률 본문과 구별되는 별도 자료 유형."""
    for el in root.findall("./제개정이유/제개정이유내용"):
        raw = re.sub(r"\n{3,}", "\n\n", (el.text or "").strip())
        for i, part in enumerate(splitter.split_text(raw) if raw else []):
            add(
                f"[{law_name}] 제개정이유\n{part}",
                {
                    "source_type": "reason",
                    "chunk_role": "child",
                    "parent_id": None,
                    "part": i,
                },
                f"{version}:REASON:{i}",
            )
