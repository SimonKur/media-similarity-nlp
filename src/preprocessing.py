import re
import pandas as pd
import glob

def parse_articles_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # 1) Remove leading header "Nyheter:" if present
    raw = re.sub(r"(?m)^\s*Nyheter:\s*\n?", "", raw)

    # 2) Split articles by the separator (===... lines)
    articles = re.split(r"=+\s*\n+", raw)
    articles = [a.strip() for a in articles if a.strip()]

    parsed = []
    # regex to strictly match "Publisher, YYYY-MM-DD" or "Publisher, YYYY-MM-DD HH:MM"
    pubdate_re = re.compile(
        r"^(?P<publisher>[\wÅÄÖåäöéÉüÜöÖæÆøØçÇ\.\-&\s]+?),\s*(?P<date>\d{4}-\d{2}-\d{2}(?:\s*\d{2}:\d{2})?)\s*$"
    )

    for art in articles:
        lines = [ln.strip() for ln in art.splitlines() if ln.strip()]
        if not lines:
            continue

        # Defaults
        title = None
        publisher = None
        date = None
        publication_type = None
        link = None
        body = ""

        # Title is first non-empty line
        title = lines[0]

        # By assumption publisher is on next non-empty line (line index 1)
        publisher_line_idx = 1 if len(lines) > 1 else None
        publisher_line = lines[publisher_line_idx] if publisher_line_idx is not None else ""

        # Try strict match on line 2
        m = pubdate_re.match(publisher_line) if publisher_line else None

        # Fallback: search within first up to 4 lines for a pub+date line
        if not m:
            for i in range(1, min(4, len(lines))):
                m2 = pubdate_re.match(lines[i])
                if m2:
                    publisher_line_idx = i
                    m = m2
                    break

        if m:
            publisher = m.group("publisher").strip()
            date = m.group("date").strip()
        else:
            # If still not found, leave publisher/date None (we can try looser parsing)
            publisher = None
            date = None

        # Publication type - search for 'Publicerat på webb' or 'Publicerat i print' anywhere
        pubtype_match = re.search(r"Publicerat\s+(på|i)\s+(\w+)", art)
        if pubtype_match:
            publication_type = f"{pubtype_match.group(1)} {pubtype_match.group(2)}"
            # find index of that line to determine where body begins
            # try to find exact line index if possible
            for idx, ln in enumerate(lines):
                if re.search(r"Publicerat\s+(på|i)\s+(\w+)", ln):
                    pubtype_idx = idx
                    break
            else:
                pubtype_idx = None
        else:
            pubtype_idx = None

        # Link: search for first http(...) occurrence
        link_match = re.search(r"(https?://[^\s]+)", art)
        if link_match:
            link = link_match.group(1).strip()

        # Body extraction logic:
        # If we detected publication type line, body begins after it.
        # Else if we detected publisher_line_idx, body begins after that.
        # Else body begins after title line.
        start_idx = None
        if pubtype_idx is not None:
            start_idx = pubtype_idx + 1
        elif publisher_line_idx is not None:
            start_idx = publisher_line_idx + 1
        else:
            start_idx = 1

        # Collect body lines from start_idx to before copyright or link lines
        body_lines = []
        for ln in lines[start_idx:]:
            # stop at copyright or 'Bildtext:' or 'Alla artiklar är skyddade' lines
            if ln.startswith("©") or "Alla artiklar" in ln or ln.lower().startswith("bildtext"):
                break
            # skip the 'Se webartikeln på ...' line
            if re.search(r"Se webartikeln på\s*http", ln):
                continue
            # skip standalone URLs
            if re.match(r"https?://", ln):
                continue
            body_lines.append(ln)

        # join body lines preserving paragraphs (use double newline)
        body = "\n\n".join(body_lines).strip()

        parsed.append({
            "Title": title,
            "Publisher": publisher,
            "Date": date,
            "PublicationType": publication_type,
            "Text": body,
            "Link": link
        })

    df = pd.DataFrame(parsed)
    return df

files = glob.glob("/Users/simonkurzewski/Desktop/Statistik-kand/Data/Retriever export-*.txt")

all_dfs = []

for f in files:
  print(f"Parsing {f} ...")
  df_temp = parse_articles_from_file(f)
  all_dfs.append(df_temp)

# Combine everything
df_all = pd.concat(all_dfs, ignore_index = True)

# drop duplicates
df_all = df_all.drop_duplicates(subset = ["Text"], keep = "first")

# Save file in csv
df_all.to_csv("structured_articles.csv", index = False, encoding = "utf-8-sig")
print("saved datset with ", len(df_all), "articles")
