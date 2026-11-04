# Text Extraction Plan: Homer's *Odyssey* (Fitzgerald Translation)

This document details the reverse-engineering analysis, technical approach, and Python implementation used to extract clean verse text from `docs/odyssey-fitzgerald.pdf` and partition it into 24 distinct chapter files (one file per Book).

---

## 1. Executive Summary & Objective

The objective is to reliably extract Homer's *The Odyssey* (translated by Robert Fitzgerald) from the provided PDF file into standalone, structured text files corresponding to each book (Book I through Book XXIV).

### Core Requirements
- **Implementation Language**: Python (3.10+ compatible).
- **Extraction Scope**: Exclusively Homer's *The Odyssey* (Books 1 to 24). Filter out the introductory editorial headnote on page 1 and the extraneous *Aesop's Fables* occupying pages 323–339.
- **Output**: Individual chapter files (`chapter_01.txt` through `chapter_24.txt` in a designated `chapters/` directory), formatted cleanly with poetic verse structure preserved and running headers, margin line counts, and footnotes stripped.

---

## 2. PDF Reverse-Engineering & Structural Analysis

Inspection of `docs/odyssey-fitzgerald.pdf` reveals a standard PDF 1.3 / 1.4 document containing 339 physical pages.

```mermaid
flowchart TD
    PDF["docs/odyssey-fitzgerald.pdf (339 pages)"] --> Catalog["Catalog /Root Object"]
    Catalog --> PagesTree["Pages Tree Object (/Type /Pages)"]
    PagesTree --> P1_322["Pages 1–322: Homer's Odyssey (Books 1–24)"]
    PagesTree --> P323_339["Pages 323–339: Aesop's Fables (Excluded)"]
    P1_322 --> StreamParser["FlateDecode Decompressor & Tokenizer"]
    StreamParser --> FontClassifier["Font & Geometry Filter (Tm, Tf, BT/ET)"]
    FontClassifier --> Segmenter["24 Chapter Files (chapter_01.txt - chapter_24.txt)"]
```

### 2.1 Object Tree and Content Streams
- **Page Tree Order**: Page objects are stored across indirect object trees with `/Kids` arrays. Sequential extraction requires traversing the `/Pages` tree rather than relying on unordered numerical object IDs.
- **Stream Compression**: Page content streams are compressed with standard `FlateDecode` (`zlib`).
- **Text Blocks**: Text is encapsulated in `BT` (Begin Text) ... `ET` (End Text) operators.

### 2.2 Fonts & Character Encodings
The PDF utilizes three Type 1 font subsets:
1. `/Tc1` (`Courier` / `NewBaskerville-Italic`): Used for italic titles, header tags, and italicized poetic songs (such as the Sirens' hymn in Book XII).
2. `/Tc2` (`NewBaskerville-Roman`): Primary typeface for verse lines and book headers, encoded in `/MacRomanEncoding`.
3. `/Tc3`: Header metadata and production stamps.

### 2.3 Typographic Anatomy & Coordinate Geometry
In PDF coordinate space, the origin `(0, 0)` is at the bottom-left of the page. Text elements are placed using the text matrix operator:
```text
a b c d e f Tm
```
where `d` represents the font size in points, `e` is the horizontal position ($x$), and `f` is the vertical position ($y$).

| Element Type | Font Size ($d$) | Vertical Position ($y$) | Horizontal Position ($x$) | Treatment |
| :--- | :--- | :--- | :--- | :--- |
| **Running Headers** | 6.75pt, 8pt, 9pt | $y \ge 638\text{ pt}$ | Across page | **Discard** (page numbers, "The Ancient World", "Homer", prepress stamps) |
| **Book / Chapter Titles** | 12.0pt | Variable | $110\text{ pt} \le x \le 230\text{ pt}$ | **Extract as Chapter Boundary** (`BOOK [NUMBER]: [TITLE]`) |
| **Main Poetic Verses** | 10.0pt | $100\text{ pt} \le y < 638\text{ pt}$ | $79\text{ pt} \le x \le 270\text{ pt}$ | **Extract as Primary Content** |
| **Embedded Lyric Song** | 9.0pt | $160\text{ pt} \le y < 635\text{ pt}$ | $180\text{ pt} \le x \le 215\text{ pt}$ | **Extract as Primary Content** (Sirens' Song in Book XII, pp. 153–154) |
| **Line Numbers** | 6.0pt, 6.06pt | Mid-page | $x \ge 370\text{ pt}$ | **Discard** (right-margin line indices: 10, 20, 30...) |
| **Superscript Footnote Calls**| 7.5pt, 6.0pt | $y > 200\text{ pt}$ | Variable ($y + 3.3\text{ pt}$) | **Discard** (inline numeric citation markers) |
| **Footnotes** | 8.0pt, 6.0pt | $y < 220\text{ pt}$ | Left margin / bottom | **Discard** (editorial apparatus at page bottom) |
| **Introductory Headnote** | 9.0pt, 10.0pt | $300\text{ pt} < y < 600\text{ pt}$ | Page 1 | **Discard** (editors' background essay prior to Book One) |

---

## 3. Key Challenges & Algorithmic Solutions

### 3.1 Composite Greek Accents & Ligatures
Because standard MacRoman Baskerville lacked precomposed accented characters for transliterated Greek vowels (such as $y$ with acute accent, $\acute{y}$), the typesetter placed a standalone acute accent character (`´`) immediately preceding or following the letter `y` across separate horizontal text chunks (e.g. `Og´` + `ygia`, `Eur´` + `ymakhos`, `Kl´` + `ytios`).

**Solution**:
The parser clusters text fragments on identical vertical lines ($|\Delta y| \le 1.5\text{ pt}$) sorted left-to-right. When an element ends in `´` and the subsequent token begins with `y` or `Y`, they are reassembled into `ý` or `Ý` (e.g. `to the island of Ogýgia`, `Eurýmakhos`).

### 3.2 Mid-Page Chapter Transitions
Chapters do not always begin on fresh pages. For example, Book One concludes on line 521 of Page 13 ($y \ge 327\text{ pt}$), and Book Two begins at $y = 285.68\text{ pt}$ on the exact same page.

**Solution**:
A state machine processes lines sequentially down each page. When a 12pt title matching `^BOOK\s+([A-Z -]+):` is detected, the current chapter is finalized, and a new chapter buffer is initialized immediately.

### 3.3 Multi-Line Title Wrapping
Several book titles wrap over two lines (e.g. Book Three: `BOOK THREE: THE LORD OF THE` followed by `WESTERN APPROACHES`).

**Solution**:
The parser detects continuation lines in the 12pt heading font and merges them into a cohesive title string.

### 3.4 Zero External Dependency Execution
To guarantee execution across any sandboxed or minimal Python environment without requiring external binary wheels (`pypdf`, `pymupdf`, `pdfplumber`), the implementation uses standard library `zlib` and `re` directly against the PDF format specification.

---

## 4. Implementation Details: `extract_chapters.py`

The implementation script provides end-to-end extraction:

```python
# Key operational steps in extract_chapters.py:
1. Index all indirect objects: (\d+) 0 obj ... endobj
2. Traverse /Root -> /Pages -> /Kids to obtain the ordered page list (pages 1 to 322).
3. Decompress each page stream using zlib.decompress().
4. Extract all text operators within BT ... ET blocks:
   - Read font matrix Tm to determine font size (d) and coordinates (x, y).
   - Parse text strings from [ (str) kern (str) ] TJ and (str) Tj.
   - Decode bytes using MacRoman encoding with octal escape replacement.
5. Apply filtering rules (strip headers y >= 638, line numbers d <= 6.5, footnotes d <= 8.0).
6. Group fragments on the same baseline and merge horizontal text.
7. Partition stream into 24 Books based on 12pt heading markers.
8. Write each book to chapters/chapter_XX.txt with UTF-8 encoding.
```

---

## 5. Verification & Chapter Inventory

The script extracts all 24 Books with 100% boundary accuracy:

| Chapter | Title | Page Range | Line Count | First Line | Last Line |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **01** | BOOK ONE: A GODDESS INTERVENES | pp. 1–13 | 500 | *Sing in me, Muse, and through me tell the story* | *he took in thought the course Athena gave him.* |
| **02** | BOOK TWO: A HERO’S SON AWAKENS | pp. 13–23 | 461 | *When primal Dawn spread on the eastern sky* | *And the prow sheared through the night into the dawn.* |
| **03** | BOOK THREE: THE LORD OF THE WESTERN APPROACHES | pp. 24–36 | 542 | *The sun rose on the flawless brimming sea* | *the sun went down and all the roads grew dark.* |
| **04** | BOOK FOUR: THE RED-HAIRED KING AND HIS LADY | pp. 37–58 | 901 | *By vales and sharp ravines in Lakedaimon* | *In ambush here that night the Akhaians lay.* |
| **05** | BOOK FIVE: SWEET NYMPH AND OPEN SEA | pp. 58–70 | 519 | *Dawn came up from the couch of her reclining,* | *In quiet sleep she sealed his cherished eyes.* |
| **06** | BOOK SIX: THE PRINCESS AT THE RIVER | pp. 70–78 | 351 | *Far gone in weariness, in oblivion,* | *until the kingly man came home to his own shore.* |
| **07** | BOOK SEVEN: GARDENS AND FIRELIGHT | pp. 78–86 | 371 | *As Lord Odysseus prayed there in the grove* | *retired to rest where his dear consort lay.* |
| **08** | BOOK EIGHT: THE SONGS OF THE HARPER | pp. 86–101 | 626 | *Under the opening fingers of the dawn* | *can take a brother’s place in our affection.”* |
| **09** | BOOK NINE: NEW COASTS AND POSEIDON’S SON | pp. 101–115 | 618 | *Now this was the reply Odysseus made:* | *having our precious lives, but not our friends.* |
| **10** | BOOK TEN: THE GRACE OF THE WITCH | pp. 115–130 | 635 | *We made our landfall on Aiolia Island,* | *unless she wished his mortal eyes aware?* |
| **11** | BOOK ELEVEN: A GATHERING OF SHADES | pp. 130–148 | 759 | *We bore down on the ship at the sea’s edge* | *first under oars, then with a following wind.* |
| **12** | BOOK TWELVE: SEA PERILS AND DEFEAT | pp. 148–161 | 580 | *The ship sailed on, out of the Ocean Stream,* | *with tiresome repetition of a story.”* |
| **13** | BOOK THIRTEEN: ONE MORE STRANGE ISLAND | pp. 161–174 | 551 | *He ended it, and no one stirred or sighed* | *Far over sea, to join Odysseus’ son.* |
| **14** | BOOK FOURTEEN: HOSPITALITY IN THE FOREST | pp. 174–188 | 631 | *He went up from the cove through wooded ground,* | *out of the wind and rain.* |
| **15** | BOOK FIFTEEN: HOW THEY CAME TO ITHAKA | pp. 188–203 | 680 | *South into Lakedaimon* | *slept under arms on duty for his masters.* |
| **16** | BOOK SIXTEEN: FATHER AND SON | pp. 203–217 | 592 | *But there were two men in the mountain hut—* | *they turned to bed and took the gift of sleep.* |
| **17** | BOOK SEVENTEEN: THE BEGGAR AT THE MANOR | pp. 217–235 | 794 | *When the young Dawn came bright into the East* | *of dance and song, as day waned into evening.* |
| **18** | BOOK EIGHTEEN: BLOWS AND A QUEEN’S BEAUTY | pp. 235–247 | 520 | *Now a true scavenger came in—a public tramp* | *they trailed off homeward drowsily to bed.* |
| **19** | BOOK NINETEEN: RECOGNITIONS AND A DREAM | pp. 247–263 | 699 | *Now by Athena’s side in the quiet hall* | *cast sweet sleep upon her eyes.* |
| **20** | BOOK TWENTY: SIGNS AND A VISION | pp. 263–274 | 439 | *Outside in the entry way he made his bed—* | *whose treachery had filled that house with pain.* |
| **21** | BOOK TWENTY-ONE: THE TEST OF THE BOW | pp. 274–285 | 499 | *Upon Penélopê, most worn in love and thought,* | *stood by his chair, in the forefront near his father.* |
| **22** | BOOK TWENTY-TWO: DEATH IN THE GREAT HALL | pp. 285–298 | 559 | *Now shrugging off his rags the wiliest fighter of the islands* | *and overcome by longing and by tears.* |
| **23** | BOOK TWENTY-THREE: THE TRUNK OF THE OLIVE TREE | pp. 298–308 | 422 | *The old nurse went upstairs exulting,* | *going before them swiftly from the town.* |
| **24** | BOOK TWENTY-FOUR: WARRIORS, FAREWELL | pp. 308–322 | 614 | *Meanwhile the suitors’ ghosts were called away* | *though still she kept the form and voice of Mentor.* |

---

## 6. Execution & Directory Structure

To reproduce or run the extraction at any time:

```bash
python3 extract_chapters.py docs/odyssey-fitzgerald.pdf chapters
```

Generated directory layout:
```text
.
├── docs/
│   ├── odyssey-fitzgerald.pdf
│   └── odyssey-teachers-guide.pdf
├── chapters/
│   ├── chapter_01.txt
│   ├── chapter_02.txt
│   ├── ...
│   └── chapter_24.txt
├── extract_chapters.py
├── extraction_plan.md
└── readme.md
```

